import time
import random
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for safety

    def is_regular_subset(adj, subset):
        """Check if induced subgraph on subset is regular."""
        k = len(subset)
        if k < 2:
            return True
        # Fast check using numpy
        indices = np.array(subset)
        sub_adj = adj[np.ix_(indices, indices)]
        degrees = np.sum(sub_adj, axis=1)
        return np.all(degrees == degrees[0])

    def verify_full(n, adj):
        """Check all subsets of size >= 6."""
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                if is_regular_subset(adj, comb):
                    return False
        return True

    def prepare_data(n):
        """Precompute combinations and edge-to-comb mappings."""
        combs = list(itertools.combinations(range(n), 6))
        edge_to_combs = {}
        for idx, comb in enumerate(combs):
            for i in range(6):
                for j in range(i + 1, 6):
                    u, v = comb[i], comb[j]
                    if u > v:
                        u, v = v, u
                    if (u, v) not in edge_to_combs:
                        edge_to_combs[(u, v)] = []
                    edge_to_combs[(u, v)].append(idx)
        return combs, edge_to_combs

    def initialize_state(n, combs):
        """Initialize adjacency matrix, degrees per subset, and regularity flags."""
        adj = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        num_combs = len(combs)
        # sub_degrees[sub_idx][k] is the degree of the k-th vertex in that subset
        sub_degrees = np.zeros((num_combs, 6), dtype=np.int8)
        is_reg = np.zeros(num_combs, dtype=bool)
        
        for idx, comb in enumerate(combs):
            indices = np.array(comb)
            sub_adj = adj[np.ix_(indices, indices)]
            degs = np.sum(sub_adj, axis=1)
            sub_degrees[idx] = degs
            is_reg[idx] = np.all(degs == degs[0])
            
        bad_count = np.sum(is_reg)
        return adj, sub_degrees, is_reg, bad_count

    def simulated_annealing(n, combs, edge_to_combs, time_limit):
        adj, sub_degrees, is_reg, bad_count = initialize_state(n, combs)
        
        if bad_count == 0:
            if verify_full(n, adj):
                return adj

        best_adj = adj.copy()
        best_bad = bad_count
        current_temp = 1.0
        cooling = 0.999
        
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        while time.time() < time_limit:
            if bad_count == 0:
                if verify_full(n, adj):
                    return adj
                # If bad_count is 0 but full verification fails, 
                # it means there's a regular subgraph of size > 6.
                # We should try to break it.
                # For now, we continue SA to hopefully break it by chance 
                # or we can specifically target it.
                # Let's just continue, as larger regular subgraphs are rare 
                # and often broken by local moves that don't increase 6-bad count.
                pass

            u, v = random.choice(edges)
            idxs = edge_to_combs.get((u, v), [])
            if not idxs:
                continue
            
            # Calculate delta
            delta = 0
            old_states = [] # Store (idx, old_reg) to rollback or update
            
            for idx in idxs:
                old_reg = is_reg[idx]
                # Get positions of u and v in comb
                comb = combs[idx]
                pos_u = comb.index(u)
                pos_v = comb.index(v)
                
                # Update degrees
                if adj[u, v] == 0:
                    # Adding edge
                    sub_degrees[idx, pos_u] += 1
                    sub_degrees[idx, pos_v] += 1
                else:
                    # Removing edge
                    sub_degrees[idx, pos_u] -= 1
                    sub_degrees[idx, pos_v] -= 1
                
                # Check new regularity
                # We only need to check if all are equal.
                # Since only two changed, we can check:
                # 1. If old_reg was True, it becomes False (degrees of u,v change by 1, others same).
                #    Wait, if old_reg was True, all degrees were d.
                #    New degrees: u,v are d+1 (or d-1), others d.
                #    Not all equal. So new_reg is always False if old_reg was True.
                # 2. If old_reg was False, it might become True.
                
                new_reg = False
                if not old_reg:
                    # Check if all degrees in sub_degrees[idx] are equal
                    # Optimization: Check if u and v degrees match the mode?
                    # Just do a simple check.
                    degs = sub_degrees[idx]
                    if degs[0] == degs[1] == degs[2] == degs[3] == degs[4] == degs[5]:
                        new_reg = True
                else:
                    new_reg = False # As derived above
                    
                if old_reg and not new_reg:
                    delta -= 1
                elif not old_reg and new_reg:
                    delta += 1
                else:
                    delta += 0
                    
                old_states.append((idx, old_reg, new_reg))

            new_bad_count = bad_count + delta
            
            # Acceptance criterion
            if delta <= 0:
                accept = True
            else:
                # Simulated Annealing
                if random.random() < np.exp(-delta / current_temp):
                    accept = True
                else:
                    accept = False
            
            if accept:
                # Apply flip
                adj[u, v] ^= 1
                adj[v, u] ^= 1
                bad_count = new_bad_count
                
                # Update is_reg array
                for idx, old_reg, new_reg in old_states:
                    is_reg[idx] = new_reg
                    
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_adj = adj.copy()
            else:
                # Rollback
                for idx, old_reg, new_reg in old_states:
                    is_reg[idx] = old_reg
                    # Degrees were modified in place, need to rollback
                    comb = combs[idx]
                    pos_u = comb.index(u)
                    pos_v = comb.index(v)
                    if adj[u, v] == 0:
                        # We tried to add, but rejected. 
                        # Wait, we modified sub_degrees BEFORE deciding to accept.
                        # If we reject, we must undo the modification.
                        # Current adj[u,v] is 0 (before flip). 
                        # We added 1 to degrees. So subtract 1.
                        sub_degrees[idx, pos_u] -= 1
                        sub_degrees[idx, pos_v] -= 1
                    else:
                        # Current adj[u,v] is 1. We subtracted 1. So add 1 back.
                        sub_degrees[idx, pos_u] += 1
                        sub_degrees[idx, pos_v] += 1

            current_temp *= cooling

        # Final check on best
        if best_bad == 0:
            if verify_full(n, best_adj):
                return best_adj
        return None

    # Strategy: Try n=21, then n=20, etc.
    # n=21 is the target for improvement.
    
    # Allocate time wisely
    # 1. Try n=21 for 60s
    # 2. Try n=20 for 40s
    # 3. Fallback to hardcoded or smaller n
    
    best_result = None
    best_n = 0
    
    # Hardcoded fallback for n=10 (very likely valid)
    def get_fallback():
        # A random graph on 10 vertices is very likely to have no regular induced subgraph of size 6.
        # Let's generate one quickly and verify.
        for _ in range(100):
            n = 10
            adj = np.zeros((n, n), dtype=np.uint8)
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1
            if verify_full(n, adj):
                return {'n': n, 'edges': [[u, v] for u in range(n) for v in range(u+1, n) if adj[u, v] == 1]}
        return {'n': 5, 'edges': []}

    fallback = get_fallback()
    best_result = fallback
    best_n = fallback['n']

    # Try n=21
    if time.time() < deadline - 60:
        combs_21, edge_to_combs_21 = prepare_data(21)
        adj_21 = simulated_annealing(21, combs_21, edge_to_combs_21, time.time() + 55)
        if adj_21 is not None:
            best_result = {'n': 21, 'edges': [[u, v] for u in range(21) for v in range(u+1, 21) if adj_21[u, v] == 1]}
            best_n = 21

    # Try n=20 if 21 failed or we have time
    if best_n < 20 and time.time() < deadline - 20:
        combs_20, edge_to_combs_20 = prepare_data(20)
        adj_20 = simulated_annealing(20, combs_20, edge_to_combs_20, time.time() + 15)
        if adj_20 is not None:
            best_result = {'n': 20, 'edges': [[u, v] for u in range(20) for v in range(u+1, 20) if adj_20[u, v] == 1]}
            best_n = 20

    return best_result

if __name__ == "__main__":
    res = construct()
    print(res['n'])
