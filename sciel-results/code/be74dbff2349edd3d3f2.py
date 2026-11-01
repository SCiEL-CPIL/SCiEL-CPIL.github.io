"""Hybrid Greedy + Local Search construction for the ternary covering code K_3(8,1)."""
# EVOLVE-BLOCK-START
import heapq
import itertools
import numpy as np
import time
import random

# Precomputation cache for the specific task
_Q = 3
_N = 8
_RADIUS = 1
_TOTAL = _Q ** _N
_WEIGHT = _Q ** np.arange(_N - 1, -1, -1)
_WORDS = None
_NEIGHBORS = None

def _init_precomputation(q, n, radius):
    global _WORDS, _NEIGHBORS, _TOTAL, _Q, _N, _WEIGHT
    if q != 3 or n != 8 or radius != 1:
        return False
    
    _Q = q
    _N = n
    _TOTAL = q ** n
    _WEIGHT = q ** np.arange(n - 1, -1, -1)
    
    # Generate all words
    _WORDS = np.empty((_TOTAL, n), dtype=np.int64)
    for i in range(_TOTAL):
        v = i
        for j in range(n - 1, -1, -1):
            _WORDS[i, j] = v % q
            v //= q
            
    # Precompute neighbors
    _NEIGHBORS = []
    for i in range(_TOTAL):
        w = _WORDS[i].copy()
        nb = [i]
        for j in range(n):
            old = w[j]
            for a in range(q):
                if a == old:
                    continue
                w[j] = a
                nb.append(int(w.dot(_WEIGHT)))
                w[j] = old
        _NEIGHBORS.append(nb)
    return True

def _greedy_init(q, n, radius):
    """Fast greedy initialization."""
    if not _init_precomputation(q, n, radius):
        # Fallback for other params
        return [list(w) for w in itertools.product(range(q), repeat=n)]
        
    uncovered = np.ones(_TOTAL, dtype=bool)
    remaining = _TOTAL
    codewords = []
    
    covered_count = np.array([len(nb) for nb in _NEIGHBORS], dtype=np.int64)
    heap = [(-covered_count[i], i) for i in range(_TOTAL)]
    heapq.heapify(heap)
    
    while remaining > 0:
        while True:
            neg_key, i = heapq.heappop(heap)
            if -neg_key == covered_count[i]:
                break
        codewords.append(_WORDS[i].tolist())
        for nb in _NEIGHBORS[i]:
            if uncovered[nb]:
                uncovered[nb] = False
                remaining -= 1
                for j in _NEIGHBORS[nb]:
                    covered_count[j] -= 1
                    if remaining > 0:
                        heapq.heappush(heap, (-covered_count[j], j))
                        
    return codewords

def construct(q, n, radius):
    if not (q == 3 and n == 8 and radius == 1):
        return [list(w) for w in itertools.product(range(q), repeat=n)]
        
    _init_precomputation(q, n, radius)
    
    # 1. Greedy Initialization
    codewords = _greedy_init(q, n, radius)
    C_indices = set(int(np.array(c, dtype=np.int64).dot(_WEIGHT)) for c in codewords)
    
    # 2. Local Search: Move + Remove
    cover_count = np.zeros(_TOTAL, dtype=np.int64)
    for idx in C_indices:
        for nb in _NEIGHBORS[idx]:
            cover_count[nb] += 1
            
    start_time = time.time()
    timeout = 110.0
    
    # Pre-compute neighbor sets for faster intersection/difference checks if needed?
    # For now, we iterate.
    
    improved = True
    while improved and (time.time() - start_time) < timeout:
        improved = False
        
        # Phase 1: Remove redundant codewords
        removed_any = False
        # We need to check if a codeword is redundant.
        # A codeword idx is redundant if min(cover_count[nb] for nb in _NEIGHBORS[idx]) > 1
        # We can iterate through current codewords.
        to_remove = []
        for idx in C_indices:
            # Check if all neighbors are covered by at least 2 codewords
            # If min cover count in neighborhood is > 1, it's safe to remove.
            if np.min(cover_count[_NEIGHBORS[idx]]) > 1:
                to_remove.append(idx)
                
        if to_remove:
            # Remove one at a time to be safe, or all at once?
            # Removing one at a time is safer for state consistency.
            idx_to_remove = to_remove[0]
            C_indices.remove(idx_to_remove)
            for nb in _NEIGHBORS[idx_to_remove]:
                cover_count[nb] -= 1
            improved = True
            removed_any = True
            continue # Restart loop to re-evaluate removals
            
        # Phase 2: If no removal, try a Move
        # Pick a random codeword to move
        if len(C_indices) > 0:
            idx_to_move = random.choice(list(C_indices))
            nb_list_move = _NEIGHBORS[idx_to_move]
            
            # We want to find a new position new_idx such that:
            # 1. new_idx is not in C_indices
            # 2. Coverage is maintained.
            # Condition: For all p in N(idx_to_move) \ N(new_idx), cover_count[p] >= 2 (so removing idx_to_move doesn't uncover p)
            # Note: We don't strictly need N(new_idx) to cover anything new, just that the union of N(C\{idx}) U N(new_idx) covers everything.
            # This is equivalent to: N(idx_to_move) \ N(new_idx) subset of {p | cover_count[p] >= 2}
            
            # Optimization: Only try moving to "good" candidates.
            # A good candidate is one that covers points currently covered only by idx_to_move?
            # Actually, if we move, we might uncover points.
            # Let's try random neighbors of idx_to_move first (local search).
            # Or try random codewords from the whole space? Too slow.
            # Let's try neighbors of idx_to_move (distance 1).
            
            candidates = _NEIGHBORS[idx_to_move]
            random.shuffle(candidates)
            
            moved = False
            for new_idx in candidates:
                if new_idx in C_indices:
                    continue
                    
                # Check validity of move
                # Points that will lose coverage: N(idx_to_move) \ N(new_idx)
                # We need cover_count[p] >= 2 for all p in this set.
                # Note: cover_count currently includes idx_to_move.
                # So if cover_count[p] == 1, it means only idx_to_move covers it.
                # If p is in N(idx_to_move) and NOT in N(new_idx), and cover_count[p] == 1, the move is invalid.
                
                # Efficient check:
                # We need to ensure no p in N(idx_to_move) \ N(new_idx) has cover_count[p] == 1.
                
                # Let's compute the set of points uniquely covered by idx_to_move.
                # unique_covered = [p for p in N(idx_to_move) if cover_count[p] == 1]
                # If any of these are NOT in N(new_idx), the move is invalid.
                
                # To make this fast, we can precompute unique coverage? No, it changes.
                # Let's just iterate N(idx_to_move).
                
                valid = True
                for p in nb_list_move:
                    if cover_count[p] == 1:
                        # This point is ONLY covered by idx_to_move.
                        # It MUST be covered by new_idx.
                        if new_idx not in _NEIGHBORS[p]: # Wait, _NEIGHBORS[p] is list of codewords covering p.
                             # Actually, _NEIGHBORS is indexed by codeword index, giving neighbor points.
                             # We need to know if new_idx covers p.
                             # Since N is symmetric for radius 1 (distance), p is in N(new_idx) iff new_idx is in N(p).
                             # But we don't have reverse mapping.
                             # However, we can just check if p is in _NEIGHBORS[new_idx].
                             # This is O(len(N)) = O(25) per check.
                             # Total checks: len(candidates) * len(N(idx)) * len(N(new_idx))?
                             # No, we can just check `p in _NEIGHBORS[new_idx]`.
                             # Let's optimize by checking if new_idx is in the "reverse neighbors" of p.
                             # We didn't precompute reverse neighbors.
                             # Let's stick to simple check.
                             pass
                             
                # Re-implementing the check properly:
                # We need: For all p in N(idx_to_move), if cover_count[p] == 1, then p must be in N(new_idx).
                
                # Let's find the set of "critical" points for idx_to_move.
                critical_points = [p for p in nb_list_move if cover_count[p] == 1]
                
                if not critical_points:
                    # Move is always valid regarding coverage (no unique points)
                    # But does it help? It might allow other removals.
                    valid = True
                else:
                    # Check if all critical points are covered by new_idx
                    valid = True
                    for p in critical_points:
                        if p not in _NEIGHBORS[new_idx]:
                            valid = False
                            break
                            
                if valid:
                    # Perform the move
                    C_indices.remove(idx_to_move)
                    for p in nb_list_move:
                        cover_count[p] -= 1
                        
                    C_indices.add(new_idx)
                    for p in _NEIGHBORS[new_idx]:
                        cover_count[p] += 1
                        
                    improved = True
                    moved = True
                    break
                    
            if moved:
                continue # Restart loop to check for removals

    return [list(_WORDS[idx]) for idx in C_indices]
# EVOLVE-BLOCK-END
