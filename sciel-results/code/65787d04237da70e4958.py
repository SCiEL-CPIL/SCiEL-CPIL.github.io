import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115
    random.seed(42)
    
    def build_adj_masks(n, edges):
        masks = [0] * n
        for u, v in edges:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        return masks

    def masks_to_edges(masks, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if masks[u] & (1 << v):
                    edges.append([u, v])
        return edges

    def is_regular_subset(subset, masks, subset_mask):
        # subset is tuple of indices
        # subset_mask is int mask of subset
        degs = []
        for u in subset:
            deg = (masks[u] & subset_mask).bit_count() - 1
            degs.append(deg)
        return len(set(degs)) == 1

    def check_full_validity(masks, n):
        # Check k=6
        for subset in combinations(range(n), 6):
            sm = 0
            for u in subset:
                sm |= (1 << u)
            if is_regular_subset(subset, masks, sm):
                return False
        # Check k > 6
        for k in range(7, n + 1):
            for subset in combinations(range(n), k):
                sm = 0
                for u in subset:
                    sm |= (1 << u)
                if is_regular_subset(subset, masks, sm):
                    return False
        return True

    def local_search(n, time_limit):
        # Precompute subsets of size 6
        subsets_6 = list(combinations(range(n), 6))
        subset_masks_6 = []
        for s in subsets_6:
            m = 0
            for u in s:
                m |= (1 << u)
            subset_masks_6.append(m)
            
        # Map vertex to list of subset indices containing it
        vert_to_subsets = [[] for _ in range(n)]
        for i, s in enumerate(subsets_6):
            for u in s:
                vert_to_subsets[u].append(i)

        # Init random graph
        masks = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)

        # Count bad 6-subsets
        num_bad = 0
        bad_flags = [False] * len(subsets_6)
        for i in range(len(subsets_6)):
            if is_regular_subset(subsets_6[i], masks, subset_masks_6[i]):
                num_bad += 1
                bad_flags[i] = True

        best_masks = masks[:]
        best_num_bad = num_bad
        T = 1.0
        cooling = 0.999
        current_time = time.time()
        
        while time.time() < current_time + time_limit:
            if num_bad == 0:
                # Check full validity
                if check_full_validity(masks, n):
                    return masks_to_edges(masks, n)
                # If not valid, we might need to continue searching to break larger subgraphs
                # But for now, assume 0 bad 6-subsets is a strong signal
                # We can try a few more steps to break larger ones if we track them, 
                # but for simplicity, we just return if valid.
                # If invalid, we continue searching, hoping to find a different 0-bad-6 graph that is fully valid.
                pass

            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
                
            # Identify affected subsets
            affected_indices = set(vert_to_subsets[u]) | set(vert_to_subsets[v])
            
            # Calculate delta
            delta = 0
            for idx in affected_indices:
                was_bad = bad_flags[idx]
                # Simulate flip
                masks[u] ^= (1 << v)
                masks[v] ^= (1 << u)
                is_bad_now = is_regular_subset(subsets_6[idx], masks, subset_masks_6[idx])
                # Revert for now
                masks[u] ^= (1 << v)
                masks[v] ^= (1 << u)
                
                if was_bad and not is_bad_now:
                    delta -= 1
                elif not was_bad and is_bad_now:
                    delta += 1
            
            new_num_bad = num_bad + delta
            if new_num_bad < num_bad:
                # Accept
                masks[u] ^= (1 << v)
                masks[v] ^= (1 << u)
                for idx in affected_indices:
                    bad_flags[idx] = is_regular_subset(subsets_6[idx], masks, subset_masks_6[idx])
                num_bad = new_num_bad
                if num_bad < best_num_bad:
                    best_num_bad = num_bad
                    best_masks = masks[:]
            else:
                # Accept with probability
                if new_num_bad > num_bad:
                    prob = 1.0 / (1.0 + (new_num_bad - num_bad))
                    if random.random() < prob:
                        masks[u] ^= (1 << v)
                        masks[v] ^= (1 << u)
                        for idx in affected_indices:
                            bad_flags[idx] = is_regular_subset(subsets_6[idx], masks, subset_masks_6[idx])
                        num_bad = new_num_bad
                else:
                    # Equal, accept
                    masks[u] ^= (1 << v)
                    masks[v] ^= (1 << u)
                    for idx in affected_indices:
                        bad_flags[idx] = is_regular_subset(subsets_6[idx], masks, subset_masks_6[idx])
                    num_bad = new_num_bad
            
            T *= cooling
            
        if best_num_bad == 0:
            if check_full_validity(best_masks, n):
                return masks_to_edges(best_masks, n)
        return None

    # Try n from 20 down to 10
    for n in range(20, 9, -1):
        time_left = deadline - time.time()
        if time_left < 10:
            break
        # Allocate time based on n
        time_alloc = min(30, time_left - 5)
        edges = local_search(n, time_alloc)
        if edges is not None:
            return {'n': n, 'edges': edges}
            
    # Fallback
    return {'n': 5, 'edges': []}
