import time
import random
import sys

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

def _build_fast_edges():
    """Build edges with precomputed vertex lists for faster iteration."""
    raw_edges = _build_edges()
    fast_edges = []
    for e in raw_edges:
        verts = sorted(e)
        fast_edges.append(verts)
    return fast_edges

def construct():
    t_start = time.time()
    deadline = t_start + 60.0 - DEADLINE_MARGIN

    edges = _build_edges()
    num_edges = len(edges)
    
    vert_to_edges = {v: [] for v in range(1, N + 1)}
    for i, e in enumerate(edges):
        for v in e:
            vert_to_edges[v].append(i)
            
    # Precompute edge sizes for quick access
    edge_sizes = [len(e) for e in edges]

    def _can_add(v, edge_counts):
        for ei in vert_to_edges[v]:
            if edge_counts[ei] == edge_sizes[ei] - 1:
                return False
        return True

    def _update_counts(v, delta, edge_counts):
        for ei in vert_to_edges[v]:
            edge_counts[ei] += delta

    def _init_counts(cur_set):
        """Build edge counts from scratch for a given set."""
        counts = [0] * num_edges
        for v in cur_set:
            _update_counts(v, 1, counts)
        return counts

    best_set = set()
    best_size = 0
    
    all_vertices = list(range(1, N + 1))
    
    def _greedy_add_all(cur_set, cur_edge_counts):
        """Add all possible missing vertices greedily. Returns count added."""
        added = 0
        missing = [v for v in all_vertices if v not in cur_set]
        random.shuffle(missing)
        for v in missing:
            if _can_add(v, cur_edge_counts):
                cur_set.add(v)
                _update_counts(v, 1, cur_edge_counts)
                added += 1
        return added

    def _try_swaps(cur_set, cur_edge_counts, cur_size, max_tries=20):
        """Try random multi-vertex swaps to improve the set.
        Returns (new_set, new_counts, new_size) if improved, else None."""
        cur_list = list(cur_set)
        random.shuffle(cur_list)
        
        for attempt in range(max_tries):
            if time.time() > deadline:
                break
            
            # Pick a random number of vertices to remove (1-3)
            num_rem = random.choice([1, 2, 3])
            if num_rem > len(cur_list):
                num_rem = len(cur_list)
            
            # Pick random vertices to remove
            rem_indices = random.sample(range(len(cur_list)), num_rem)
            rem_vertices = [cur_list[i] for i in rem_indices]
            
            # Save state
            old_set = cur_set.copy()
            old_edge_counts = cur_edge_counts.copy()
            old_size = cur_size
            
            # Remove vertices
            for v in rem_vertices:
                cur_set.remove(v)
                _update_counts(v, -1, cur_edge_counts)
            
            # Try to add all possible missing vertices
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
                return cur_set, cur_edge_counts, new_size
            else:
                # Rollback
                cur_set = old_set
                cur_edge_counts = old_edge_counts
                cur_size = old_size
        
        return None

    # Diverse initial seeds
    seed_patterns = [
        lambda: set(range(65, N + 1)),           # Upper half
        lambda: set(range(85, N + 1)),           # Upper third
        lambda: set(range(40, N + 1)),           # Upper 69%
        lambda: set(v for v in range(1, N + 1) if v >= 50),  # v >= 50
        lambda: set(v for v in range(1, N + 1) if v >= 60 and v % 2 == 0) | set(range(90, N + 1)),
        lambda: set(range(70, N + 1)) | set(v for v in range(1, 30) if v % 3 == 0),
    ]
    
    num_restarts = 500
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Use diverse seeds, cycling through patterns
        if r < len(seed_patterns):
            cur_set = seed_patterns[r]()
        elif r % 3 == 0:
            cur_set = set(range(65, N + 1))
        elif r % 3 == 1:
            cur_set = set(range(75, N + 1))
        else:
            cur_set = set(range(60, N + 1))
        
        # Perturb: remove random elements (more aggressive for later restarts)
        perturb_factor = 1 + min(r // 100, 2)  # Scale up perturbations
        num_remove = random.randint(0, 12 * perturb_factor)
        if num_remove > len(cur_set):
            num_remove = len(cur_set) // 2
        for _ in range(num_remove):
            if cur_set:
                v = random.choice(list(cur_set))
                cur_set.remove(v)
        
        cur_edge_counts = _init_counts(cur_set)
        cur_size = len(cur_set)
        
        if cur_size > best_size:
            best_size = cur_size
            best_set = cur_set.copy()
            
        # Local Search with multi-swap
        stuck_count = 0
        max_stuck = 40
        while time.time() < deadline:
            improved = False
            
            # 1. Try to add all possible missing vertices
            added = _greedy_add_all(cur_set, cur_edge_counts)
            if added > 0:
                cur_size += added
                if cur_size > best_size:
                    best_size = cur_size
                    best_set = cur_set.copy()
                stuck_count = 0
                continue
                
            # 2. No adds possible. Try multi-pair swaps.
            swap_result = _try_swaps(cur_set, cur_edge_counts, cur_size, max_tries=15)
            if swap_result is not None:
                cur_set, cur_edge_counts, cur_size = swap_result
                if cur_size > best_size:
                    best_size = cur_size
                    best_set = cur_set.copy()
                improved = True
            
            if improved:
                stuck_count = 0
            else:
                stuck_count += 1
                if stuck_count > max_stuck:
                    break

    # Final Polish: Exhaustive greedy add on best_set
    if time.time() < deadline:
        be = _init_counts(best_set)
        
        # Multiple passes with different orderings
        for _ in range(8):
            missing_final = [v for v in all_vertices if v not in best_set]
            random.shuffle(missing_final)
            for v in missing_final:
                if _can_add(v, be):
                    best_set.add(v)
                    _update_counts(v, 1, be)
        
        # Try removing each vertex and re-adding to see if we can do better
        improved_final = True
        while improved_final and time.time() < deadline:
            improved_final = False
            be = _init_counts(best_set)
            
            # Try removing pairs of vertices too
            best_list = list(best_set)
            random.shuffle(best_list)
            
            for rem_v in best_list:
                if time.time() > deadline:
                    break
                # Save
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
            
            # Also try pair removals
            if not improved_final and time.time() < deadline:
                be = _init_counts(best_set)
                best_list = list(best_set)
                random.shuffle(best_list)
                
                for i in range(min(30, len(best_list))):
                    if time.time() > deadline:
                        break
                    rem_v1 = best_list[i]
                    # Find another vertex to remove
                    if i + 1 < len(best_list):
                        rem_v2 = best_list[i + 1]
                    else:
                        continue
                    
                    old_set = best_set.copy()
                    old_be = be.copy()
                    
                    best_set.remove(rem_v1)
                    _update_counts(rem_v1, -1, be)
                    best_set.remove(rem_v2)
                    _update_counts(rem_v2, -1, be)
                    
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
            be = _init_counts(best_set)
            missing_final = [v for v in all_vertices if v not in best_set]
            random.shuffle(missing_final)
            for v in missing_final:
                if _can_add(v, be):
                    best_set.add(v)
                    _update_counts(v, 1, be)
                
    result = [int(v) for v in sorted(best_set)]
    return result

if __name__ == "__main__":
    print("Starting...")
    res = construct()
    print(f"Size: {len(res)}")
