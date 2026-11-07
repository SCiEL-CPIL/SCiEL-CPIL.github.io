import time
import random
import sys
import math

N = 128
DEADLINE_MARGIN = 5.0

def _build_edges():
    edges = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 > N or s2 > N or s3 > N:
                    continue
                edge = frozenset([a, b, c, s1, s2, s3])
                edges.append(edge)
    return edges

def construct():
    t_start = time.time()
    deadline = t_start + 60.0 - DEADLINE_MARGIN

    edges = _build_edges()
    num_edges = len(edges)
    
    vert_to_edges = {v: [] for v in range(1, N + 1)}
    for i, e in enumerate(edges):
        for v in e:
            vert_to_edges[v].append(i)
            
    edge_sizes = [len(e) for e in edges]

    def _can_add(v, edge_counts):
        for ei in vert_to_edges[v]:
            if edge_counts[ei] == edge_sizes[ei] - 1:
                return False
        return True

    def _update_counts(v, delta, edge_counts):
        for ei in vert_to_edges[v]:
            edge_counts[ei] += delta

    best_set = set()
    best_size = 0
    
    all_vertices = list(range(1, N + 1))
    
    def _greedy_add_all(cur_set, cur_edge_counts):
        added = 0
        missing = [v for v in all_vertices if v not in cur_set]
        random.shuffle(missing)
        for v in missing:
            if _can_add(v, cur_edge_counts):
                cur_set.add(v)
                _update_counts(v, 1, cur_edge_counts)
                added += 1
        return added

    def _local_search(cur_set, cur_edge_counts, cur_size):
        stuck_count = 0
        max_stuck = 30
        while time.time() < deadline:
            improved = False
            
            added = _greedy_add_all(cur_set, cur_edge_counts)
            if added > 0:
                cur_size += added
                stuck_count = 0
                continue
                
            found_swap = False
            if cur_set:
                cur_list = list(cur_set)
                random.shuffle(cur_list)
                
                for num_rem in [1, 2, 3]:
                    if num_rem > len(cur_list):
                        continue
                    
                    rem_vertices = cur_list[:num_rem]
                    
                    old_set = cur_set.copy()
                    old_edge_counts = cur_edge_counts.copy()
                    old_size = cur_size
                    
                    for v in rem_vertices:
                        cur_set.remove(v)
                        _update_counts(v, -1, cur_edge_counts)
                    
                    missing_after = [v for v in all_vertices if v not in cur_set]
                    random.shuffle(missing_after)
                    
                    local_added = 0
                    for v in missing_after:
                        if _can_add(v, cur_edge_counts):
                            cur_set.add(v)
                            _update_counts(v, 1, cur_edge_counts)
                            local_added += 1
                    
                    new_size = cur_size - num_rem + local_added
                    
                    if new_size > old_size:
                        cur_size = new_size
                        improved = True
                        found_swap = True
                        break
                    else:
                        cur_set = old_set
                        cur_edge_counts = old_edge_counts
                        cur_size = old_size
                    
                    if found_swap:
                        break
            
            if improved:
                stuck_count = 0
            else:
                stuck_count += 1
                if stuck_count > max_stuck:
                    break
        
        return cur_set, cur_edge_counts, cur_size

    def _lagrangian_relaxation():
        """Lagrangian relaxation: dual variables per edge guide greedy primal."""
        nonlocal best_set, best_size
        
        # Dual variables: lambda[ei] >= 0 for each edge constraint
        # Primal: maximize sum(x_v) subject to sum_{v in edge_i} x_v <= |edge_i| - 1
        # Lagrangian: L(x, lambda) = sum(x_v) - sum_i lambda_i * (sum_{v in edge_i} x_v - (|edge_i|-1))
        #            = sum_v x_v * (1 - sum_{i: v in edge_i} lambda_i) + sum_i lambda_i * (|edge_i|-1)
        # Primal subproblem: for each v, include x_v=1 if coefficient > 0
        
        # Initialize duals
        lam = [0.0] * num_edges
        
        lr_best_set = set()
        lr_best_size = 0
        
        max_dual_iters = 200
        step_size = 0.5
        step_decay = 0.98
        
        for it in range(max_dual_iters):
            if time.time() > deadline:
                break
            
            # Primal subproblem: greedy selection based on Lagrangian coefficients
            # coeff[v] = 1 - sum_{i: v in edge_i} lam[i]
            coeff = [0.0] * (N + 1)
            for v in range(1, N + 1):
                c = 1.0
                for ei in vert_to_edges[v]:
                    c -= lam[ei]
                coeff[v] = c
            
            # Select vertices with positive coefficient, break ties randomly
            candidates = [v for v in range(1, N + 1) if coeff[v] > 0]
            random.shuffle(candidates)
            
            cur_set = set()
            cur_edge_counts = [0] * num_edges
            for v in candidates:
                if _can_add(v, cur_edge_counts):
                    cur_set.add(v)
                    _update_counts(v, 1, cur_edge_counts)
            
            cur_size = len(cur_set)
            
            # Check validity (should be valid by construction, but verify)
            valid = True
            for ei in range(num_edges):
                if cur_edge_counts[ei] == edge_sizes[ei]:
                    valid = False
                    break
            
            if valid and cur_size > lr_best_size:
                lr_best_size = cur_size
                lr_best_set = cur_set.copy()
            
            # Update dual variables (subgradient method)
            # Subgradient for lambda_i = max(0, sum_{v in edge_i} x_v - (|edge_i|-1))
            # Since our primal ensures sum <= |edge_i|-1, the violation is 0
            # But we can still use a modified update to guide exploration
            
            # Alternative: use a penalty-based update
            # If an edge is "close" to being violated (sum = |edge_i|-1), increase its lambda
            # This discourages adding vertices that would complete that edge
            violations = [0.0] * num_edges
            for ei in range(num_edges):
                if cur_edge_counts[ei] >= edge_sizes[ei] - 1:
                    violations[ei] = 1.0
                elif cur_edge_counts[ei] >= edge_sizes[ei] - 2:
                    violations[ei] = 0.5
            
            # Update lambdas
            for ei in range(num_edges):
                if violations[ei] > 0:
                    lam[ei] = min(lam[ei] + step_size * violations[ei], 2.0)
                else:
                    lam[ei] = max(lam[ei] - step_size * 0.1, 0.0)
            
            step_size *= step_decay
        
        # Final local search on best found
        if lr_best_size > 0 and time.time() < deadline:
            ls_counts = [0] * num_edges
            for v in lr_best_set:
                _update_counts(v, 1, ls_counts)
            ls_set, ls_counts, ls_size = _local_search(lr_best_set, ls_counts, lr_best_size)
            if ls_size > lr_best_size:
                lr_best_set = ls_set
                lr_best_size = ls_size
        
        if lr_best_size > best_size:
            best_size = lr_best_size
            best_set = lr_best_set.copy()
        
        return lr_best_set, lr_best_size

    def _polish(best_set):
        if time.time() >= deadline:
            return best_set
        
        be = [0] * num_edges
        for v in best_set:
            _update_counts(v, 1, be)
        
        for _ in range(5):
            missing_final = [v for v in all_vertices if v not in best_set]
            random.shuffle(missing_final)
            for v in missing_final:
                if _can_add(v, be):
                    best_set.add(v)
                    _update_counts(v, 1, be)
        
        improved_final = True
        while improved_final and time.time() < deadline:
            improved_final = False
            be = [0] * num_edges
            for v in best_set:
                _update_counts(v, 1, be)
            
            for rem_v in list(best_set):
                old_set = best_set.copy()
                old_be = be.copy()
                
                best_set.remove(rem_v)
                _update_counts(rem_v, -1, be)
                
                missing_after = [v for v in all_vertices if v not in best_set]
                random.shuffle(missing_after)
                
                added = 0
                for v in missing_after:
                    if _can_add(v, be):
                        best_set.add(v)
                        _update_counts(v, 1, be)
                        added += 1
                
                if len(best_set) > len(old_set):
                    improved_final = True
                    break
                else:
                    best_set = old_set
                    be = old_be
        
        if time.time() < deadline:
            be = [0] * num_edges
            for v in best_set:
                _update_counts(v, 1, be)
            missing_final = [v for v in all_vertices if v not in best_set]
            random.shuffle(missing_final)
            for v in missing_final:
                if _can_add(v, be):
                    best_set.add(v)
                    _update_counts(v, 1, be)
        
        return best_set

    # Phase 1: Lagrangian relaxation
    _lagrangian_relaxation()
    
    # Phase 2: Greedy restarts with local search
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r % 2 == 0:
            cur_set = set(range(65, N + 1))
        else:
            cur_set = set()
            for v in range(1, N + 1):
                if random.random() < 0.5:
                    cur_set.add(v)
        
        num_remove = random.randint(0, 10)
        for _ in range(num_remove):
            if cur_set:
                v = random.choice(list(cur_set))
                cur_set.remove(v)
        
        cur_edge_counts = [0] * num_edges
        for v in cur_set:
            _update_counts(v, 1, cur_edge_counts)
            
        cur_size = len(cur_set)
        
        if cur_size > best_size:
            best_size = cur_size
            best_set = cur_set.copy()
            
        cur_set, cur_edge_counts, cur_size = _local_search(cur_set, cur_edge_counts, cur_size)
        
        if cur_size > best_size:
            best_size = cur_size
            best_set = cur_set.copy()
    
    # Phase 3: More Lagrangian runs if time permits
    if time.time() < deadline:
        for _ in range(3):
            if time.time() > deadline:
                break
            _lagrangian_relaxation()
    
    # Final polish
    if best_set:
        best_set = _polish(best_set)
    
    result = [int(v) for v in sorted(best_set)]
    return result

if __name__ == "__main__":
    print("Starting...")
    res = construct()
    print(f"Size: {len(res)}")
