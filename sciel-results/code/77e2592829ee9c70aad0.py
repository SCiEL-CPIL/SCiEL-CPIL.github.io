import time
import random
import sys

N = 128
DEADLINE_MARGIN = 5.0

def _build_fast_edges():
    """Build edges as sorted vertex lists for fast iteration."""
    edges = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 > N or s2 > N or s3 > N:
                    continue
                verts = sorted(set([a, b, c, s1, s2, s3]))
                edges.append(verts)
    return edges

def construct():
    t_start = time.time()
    deadline = t_start + 60.0 - DEADLINE_MARGIN

    # Build edges as sorted lists
    edges = _build_fast_edges()
    num_edges = len(edges)
    
    # Build vertex-to-edge index
    vert_to_edges = [[] for _ in range(N + 1)]
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

    def _init_counts(cur_set):
        """Build edge counts from scratch for a given set."""
        counts = [0] * num_edges
        for v in cur_set:
            _update_counts(v, 1, counts)
        return counts

    best_set = set()
    best_size = 0
    
    all_vertices = list(range(1, N + 1))
    
    def _greedy_add_all(cur_set, cur_edge_counts, order=None):
        """Add all possible missing vertices greedily. Returns count added."""
        if order is None:
            order = [v for v in all_vertices if v not in cur_set]
            random.shuffle(order)
        added = 0
        for v in order:
            if v not in cur_set and _can_add(v, cur_edge_counts):
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
            
            num_rem = random.choice([1, 2, 3])
            if num_rem > len(cur_list):
                num_rem = len(cur_list)
            
            rem_indices = random.sample(range(len(cur_list)), num_rem)
            rem_vertices = [cur_list[i] for i in rem_indices]
            
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
                return cur_set, cur_edge_counts, new_size
            else:
                cur_set = old_set
                cur_edge_counts = old_edge_counts
                cur_size = old_size
        
        return None

    def _small_large_search(small_thresh):
        """Decompose into small (<=thresh) and large (>thresh) parts.
        Large-large sums exceed N, so constraints are mainly:
        - small-small-small triples where all sums are in range
        - small-small-large where the large is a sum of two smalls
        - small-large-large where large1+large2 > N (no constraint)
        This allows independent optimization of the large set given the small set.
        """
        small_candidates = [v for v in all_vertices if v <= small_thresh]
        large_candidates = [v for v in all_vertices if v > small_thresh]
        
        # For each possible small set, find best large set
        # Start with greedy small set, then optimize large
        best_local = set()
        best_local_size = 0
        
        # Try different small set sizes
        for small_frac in [0.2, 0.3, 0.4, 0.5, 0.6]:
            if time.time() > deadline:
                break
            target_small = int(len(small_candidates) * small_frac)
            
            # Greedy build small set
            small_set = set()
            small_counts = [0] * num_edges
            
            # Add small elements greedily
            small_order = list(small_candidates)
            random.shuffle(small_order)
            for v in small_order:
                if len(small_set) >= target_small:
                    break
                if _can_add(v, small_counts):
                    small_set.add(v)
                    _update_counts(v, 1, small_counts)
            
            # Now find best large set given small_set
            # Large elements only conflict with each other if their pair-sum is also large
            # But large+large > N always, so large elements only conflict via small-small-large
            # A large element L is blocked if there exist small a,b in small_set with a+b=L or a+L in small_set or b+L in small_set
            # Actually: for triple (a,b,c) with a<b<c all in set, need a+b, a+c, b+c all in set
            # If c is large and a,b are small: a+b might be small or large, a+c > N (if c large enough), b+c > N
            # So the constraint for large c with small a,b is: a+b must NOT be in set (since a+c and b+c > N)
            # Wait, that's not right. The constraint is: NO triple a,b,c where ALL THREE sums are in set.
            # So if a+c > N, then a+c is not in set, so the triple is automatically safe.
            # Therefore large elements are only constrained by small-small pairs where the sum is also small.
            
            # For a large element L to be addable:
            # It must not form a violating triple with any two small elements
            # Triple (a, b, L) where a,b small, L large:
            #   Sums: a+b (small or large), a+L (>N if L large enough), b+L (>N)
            #   If a+L > N and b+L > N, then only a+b matters. If a+b in set, we have a problem only if...
            #   Wait, the constraint is that a+b, a+L, b+L are ALL in set. If a+L > N, it's not in set, so safe.
            # So large elements are safe to add unless there's a triple where all three sums are in range.
            
            # Let me just use the general _can_add check
            large_set = set()
            large_counts = small_counts.copy()  # Start with small set's counts
            
            # Add large elements greedily
            large_order = list(large_candidates)
            random.shuffle(large_order)
            for v in large_order:
                if _can_add(v, large_counts):
                    large_set.add(v)
                    _update_counts(v, 1, large_counts)
            
            full_set = small_set | large_set
            full_size = len(full_set)
            
            if full_size > best_local_size:
                best_local_size = full_size
                best_local = full_set.copy()
        
        return best_local, best_local_size

    # Phase 1: Small-large decomposition search
    for thresh in [40, 50, 60, 70]:
        if time.time() > deadline:
            break
        local_set, local_size = _small_large_search(thresh)
        if local_size > best_size:
            best_size = local_size
            best_set = local_set.copy()

    # Phase 2: Random restart local search
    seed_patterns = [
        lambda: set(range(65, N + 1)),
        lambda: set(range(85, N + 1)),
        lambda: set(range(40, N + 1)),
        lambda: set(v for v in range(1, N + 1) if v >= 50),
        lambda: set(range(70, N + 1)) | set(v for v in range(1, 30) if v % 3 == 0),
    ]
    
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r < len(seed_patterns):
            cur_set = seed_patterns[r]()
        else:
            cur_set = set(range(random.randint(50, 70), N + 1))
        
        # Perturb
        num_remove = random.randint(0, 8)
        if num_remove > len(cur_set):
            num_remove = len(cur_set) // 2
        cur_list = list(cur_set)
        random.shuffle(cur_list)
        for v in cur_list[:num_remove]:
            cur_set.remove(v)
        
        cur_edge_counts = _init_counts(cur_set)
        cur_size = len(cur_set)
        
        if cur_size > best_size:
            best_size = cur_size
            best_set = cur_set.copy()
            
        # Local search
        stuck_count = 0
        max_stuck = 30
        while time.time() < deadline:
            improved = False
            
            added = _greedy_add_all(cur_set, cur_edge_counts)
            if added > 0:
                cur_size += added
                if cur_size > best_size:
                    best_size = cur_size
                    best_set = cur_set.copy()
                stuck_count = 0
                continue
                
            swap_result = _try_swaps(cur_set, cur_edge_counts, cur_size, max_tries=10)
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

    # Final Polish
    if time.time() < deadline:
        be = _init_counts(best_set)
        
        for _ in range(10):
            missing_final = [v for v in all_vertices if v not in best_set]
            random.shuffle(missing_final)
            for v in missing_final:
                if _can_add(v, be):
                    best_set.add(v)
                    _update_counts(v, 1, be)
        
        # Try removing each vertex and re-adding
        improved_final = True
        while improved_final and time.time() < deadline:
            improved_final = False
            be = _init_counts(best_set)
            best_list = list(best_set)
            random.shuffle(best_list)
            
            for rem_v in best_list:
                if time.time() > deadline:
                    break
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
            
            # Try pair removals
            if not improved_final and time.time() < deadline:
                be = _init_counts(best_set)
                best_list = list(best_set)
                random.shuffle(best_list)
                
                for i in range(min(20, len(best_list) - 1)):
                    if time.time() > deadline:
                        break
                    rem_v1 = best_list[i]
                    rem_v2 = best_list[i + 1]
                    
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
