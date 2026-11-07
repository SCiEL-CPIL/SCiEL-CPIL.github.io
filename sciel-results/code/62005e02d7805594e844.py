import time
import random
import sys

N = 128
DEADLINE_MARGIN = 5.0  # seconds to leave before 60s limit

def _build_edges():
    """
    Build the set of forbidden configurations.
    """
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
    
    # Map vertex -> list of edge indices containing it
    vert_to_edges = {v: [] for v in range(1, N + 1)}
    for i, e in enumerate(edges):
        for v in e:
            vert_to_edges[v].append(i)
    
    # Precompute edge sizes
    edge_sizes = [len(e) for e in edges]
    
    # Dual variables (Lagrangian multipliers) for each edge
    lambda_ = [0.0] * num_edges
    
    best_set = set()
    best_size = 0
    
    def _greedy_primal(lam, seed=42):
        """Greedy primal: add vertices in order of Lagrangian score."""
        rng = random.Random(seed)
        selected = set()
        edge_counts = [0] * num_edges
        remaining = set(range(1, N + 1))
        
        # Initial scores
        scores = {}
        for v in range(1, N + 1):
            score = 1.0
            for ei in vert_to_edges[v]:
                if edge_counts[ei] == edge_sizes[ei] - 1:
                    score -= lam[ei]
            scores[v] = score
        
        while remaining:
            # Find max score vertex with random tie-breaking
            best_v = max(remaining, key=lambda v: (scores[v], rng.random()))
            
            if scores[best_v] <= 0:
                break
            
            # Add best_v
            selected.add(best_v)
            remaining.discard(best_v)
            
            # Update edge counts
            for ei in vert_to_edges[best_v]:
                edge_counts[ei] += 1
            
            # Update scores of vertices sharing edges with best_v
            affected = set()
            for ei in vert_to_edges[best_v]:
                for v in edges[ei]:
                    if v in remaining:
                        affected.add(v)
            
            for v in affected:
                score = 1.0
                for ei in vert_to_edges[v]:
                    if edge_counts[ei] == edge_sizes[ei] - 1:
                        score -= lam[ei]
                scores[v] = score
        
        return selected
    
    # Subgradient update
    def _subgradient_update(lam, selected, step_size):
        """Update dual variables based on violations."""
        new_lam = lam[:]
        for i, e in enumerate(edges):
            if all(v in selected for v in e):
                new_lam[i] += step_size
            else:
                new_lam[i] = max(0.0, new_lam[i] - step_size * 0.1)
        return new_lam
    
    # Main Lagrangian loop
    random.seed(42)
    num_lagrangian_iters = 200
    base_step = 1.0
    
    for iter_num in range(num_lagrangian_iters):
        if time.time() > deadline:
            break
        
        # Greedy primal
        cur_set = _greedy_primal(lambda_, seed=iter_num % 100)
        cur_size = len(cur_set)
        
        if cur_size > best_size:
            best_size = cur_size
            best_set = cur_set.copy()
        
        # Compute step size
        violations = 0
        for e in edges:
            if all(v in cur_set for v in e):
                violations += 1
        
        if violations == 0:
            break
        
        step_size = base_step / (1.0 + iter_num * 0.1)
        lambda_ = _subgradient_update(lambda_, cur_set, step_size)
    
    # Local search polish on best_set
    if time.time() < deadline:
        best_edge_counts = [0] * num_edges
        for v in best_set:
            for ei in vert_to_edges[v]:
                best_edge_counts[ei] += 1
        
        cur_set = best_set
        cur_edge_counts = best_edge_counts
        cur_size = len(cur_set)
        all_vertices = set(range(1, N + 1))
        
        # Try to add any valid vertex
        for v in sorted(all_vertices - cur_set):
            can_add = True
            for ei in vert_to_edges[v]:
                if cur_edge_counts[ei] == edge_sizes[ei] - 1:
                    can_add = False
                    break
            if can_add:
                cur_set.add(v)
                for ei in vert_to_edges[v]:
                    cur_edge_counts[ei] += 1
                cur_size += 1
        
        if cur_size > best_size:
            best_size = cur_size
            best_set = cur_set.copy()
        
        # Local search: try removals and re-adds
        for ls_iter in range(500):
            if time.time() > deadline:
                break
            
            improved = False
            missing = list(all_vertices - cur_set)
            random.shuffle(missing)
            
            for v in missing:
                can_add = True
                for ei in vert_to_edges[v]:
                    if cur_edge_counts[ei] == edge_sizes[ei] - 1:
                        can_add = False
                        break
                if can_add:
                    cur_set.add(v)
                    for ei in vert_to_edges[v]:
                        cur_edge_counts[ei] += 1
                    cur_size += 1
                    improved = True
            
            if improved:
                if cur_size > best_size:
                    best_size = cur_size
                    best_set = cur_set.copy()
                continue
            
            # Try removing a random vertex and re-adding
            if cur_set:
                rem_v = random.choice(list(cur_set))
                old_set = cur_set.copy()
                old_edge_counts = cur_edge_counts.copy()
                old_size = cur_size
                
                cur_set.remove(rem_v)
                for ei in vert_to_edges[rem_v]:
                    cur_edge_counts[ei] -= 1
                cur_size -= 1
                
                missing_after = list(all_vertices - cur_set)
                random.shuffle(missing_after)
                for v in missing_after:
                    if v == rem_v:
                        continue
                    can_add = True
                    for ei in vert_to_edges[v]:
                        if cur_edge_counts[ei] == edge_sizes[ei] - 1:
                            can_add = False
                            break
                    if can_add:
                        cur_set.add(v)
                        for ei in vert_to_edges[v]:
                            cur_edge_counts[ei] += 1
                        cur_size += 1
                
                if cur_size > old_size:
                    if cur_size > best_size:
                        best_size = cur_size
                        best_set = cur_set.copy()
                else:
                    cur_set = old_set
                    cur_edge_counts = old_edge_counts
                    cur_size = old_size
    
    result = [int(v) for v in sorted(best_set)]
    return result

if __name__ == "__main__":
    print("Starting...")
    res = construct()
    print(f"Size: {len(res)}")
