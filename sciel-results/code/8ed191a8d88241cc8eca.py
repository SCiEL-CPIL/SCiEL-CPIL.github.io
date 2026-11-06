import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    N = 16
    ALL_TRIPLES = list(combinations(range(N), 3))
    NUM_TRIPLES = len(ALL_TRIPLES)
    
    # Precompute pair masks for faster linearity checks
    # pair_mask[p1][p2] = 1 << pair_index, or we just use a set of frozensets
    # For speed, use a set of (min, max) tuples
    
    def triple_to_mask(t):
        """Convert a triple (a,b,c) to a bitmask."""
        return (1 << t[0]) | (1 << t[1]) | (1 << t[2])
    
    def compute_alpha_fast(triple_masks):
        """
        Fast alpha computation using precomputed 'contains triple' table.
        """
        if not triple_masks:
            return N, 0
        
        contains = bytearray(1 << N)
        
        for mask in triple_masks:
            complement = (~mask) & ((1 << N) - 1)
            extra = complement
            while True:
                s = mask | extra
                contains[s] = 1
                if extra == 0:
                    break
                extra = (extra - 1) & complement
        
        best = 0
        best_set = 0
        for s in range(1, 1 << N):
            if contains[s] == 0:
                bc = s.bit_count()
                if bc > best:
                    best = bc
                    best_set = s
        return best, best_set
    
    def compute_alpha_only(triple_masks):
        """Compute only the alpha value, not the set."""
        alpha, _ = compute_alpha_fast(triple_masks)
        return alpha
    
    def is_valid_triple(triple, used_pairs):
        """Check if adding this triple would violate linearity."""
        a, b, c = triple
        if (a, b) in used_pairs or (b, a) in used_pairs:
            return False
        if (a, c) in used_pairs or (c, a) in used_pairs:
            return False
        if (b, c) in used_pairs or (c, b) in used_pairs:
            return False
        return True
    
    def add_triple(triples, triple_masks, used_pairs, t):
        """Add a triple to the structures."""
        a, b, c = t
        used_pairs.add((a, b))
        used_pairs.add((b, a))
        used_pairs.add((a, c))
        used_pairs.add((c, a))
        used_pairs.add((b, c))
        used_pairs.add((c, b))
        triples.append(t)
        triple_masks.append(triple_to_mask(t))
    
    def remove_triple(triples, triple_masks, used_pairs, idx):
        """Remove a triple from the structures."""
        t = triples[idx]
        a, b, c = t
        used_pairs.discard((a, b))
        used_pairs.discard((b, a))
        used_pairs.discard((a, c))
        used_pairs.discard((c, a))
        used_pairs.discard((b, c))
        used_pairs.discard((c, b))
        triples.pop(idx)
        triple_masks.pop(idx)
    
    def greedy_construct(seed_val):
        """Greedy construction with given seed."""
        rng = random.Random(seed_val)
        all_t = list(ALL_TRIPLES)
        rng.shuffle(all_t)
        
        used_pairs = set()
        triples = []
        triple_masks = []
        
        for t in all_t:
            if is_valid_triple(t, used_pairs):
                add_triple(triples, triple_masks, used_pairs, t)
        
        return triples, triple_masks, used_pairs
    
    def local_search(triples, triple_masks, used_pairs, max_iterations=20000, time_limit=None):
        """
        Local search: try to improve by adding, removing, or swapping triples.
        Goal: minimize independence number.
        """
        triples = list(triples)
        triple_masks = list(triple_masks)
        used_pairs = set(used_pairs)
        
        current_alpha, current_best_set = compute_alpha_fast(triple_masks)
        
        improved = True
        iters = 0
        while improved and iters < max_iterations:
            if time_limit is not None and time.time() - start_time > time_limit:
                break
            improved = False
            iters += 1
            
            # Get the independent set to target
            indep_points = set()
            s = current_best_set
            for p in range(N):
                if s & (1 << p):
                    indep_points.add(p)
            
            # Strategy 1: Add a triple that intersects the independent set
            candidates = []
            for t in ALL_TRIPLES:
                if t not in triples and is_valid_triple(t, used_pairs):
                    candidates.append(t)
            
            if candidates:
                # Prioritize triples that intersect the independent set
                intersecting = [t for t in candidates if any(p in indep_points for p in t)]
                non_intersecting = [t for t in candidates if not any(p in indep_points for p in t)]
                
                to_try = intersecting + non_intersecting
                rng = random.Random(iters * 31 + 7)
                rng.shuffle(to_try)
                
                for t in to_try[:min(15, len(to_try))]:
                    add_triple(triples, triple_masks, used_pairs, t)
                    
                    new_alpha, new_best_set = compute_alpha_fast(triple_masks)
                    
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        current_best_set = new_best_set
                        improved = True
                    else:
                        # Revert
                        remove_triple(triples, triple_masks, used_pairs, len(triples) - 1)
                    
                    if time_limit is not None and time.time() - start_time > time_limit:
                        break
            
            # Strategy 2: Swap - remove a triple, add a better one
            if len(triples) > 0 and (time_limit is None or time.time() - start_time < time_limit):
                rng2 = random.Random(iters * 17 + 3)
                for _ in range(min(8, len(triples))):
                    idx = rng2.randrange(len(triples))
                    old_t = triples[idx]
                    remove_triple(triples, triple_masks, used_pairs, idx)
                    
                    # Find valid triples to add, prioritizing those hitting indep_points
                    candidates2 = []
                    for t in ALL_TRIPLES:
                        if t not in triples and is_valid_triple(t, used_pairs):
                            candidates2.append(t)
                    
                    intersecting2 = [t for t in candidates2 if any(p in indep_points for p in t)]
                    non_intersecting2 = [t for t in candidates2 if not any(p in indep_points for p in t)]
                    to_try2 = intersecting2 + non_intersecting2
                    
                    if to_try2:
                        rng2.shuffle(to_try2)
                        new_t = to_try2[0]
                        add_triple(triples, triple_masks, used_pairs, new_t)
                        
                        new_alpha, new_best_set = compute_alpha_fast(triple_masks)
                        
                        if new_alpha < current_alpha:
                            current_alpha = new_alpha
                            current_best_set = new_best_set
                            improved = True
                        else:
                            # Revert swap
                            remove_triple(triples, triple_masks, used_pairs, len(triples) - 1)
                            # Restore old triple
                            add_triple(triples, triple_masks, used_pairs, old_t)
                    else:
                        # Restore old triple
                        add_triple(triples, triple_masks, used_pairs, old_t)
                    
                    if time_limit is not None and time.time() - start_time > time_limit:
                        break
            
            # Strategy 3: Remove a triple that doesn't help (reduce size if alpha doesn't improve)
            # Actually, we want to keep triples that reduce alpha. But if we have too many,
            # we might remove ones that don't contribute. For now, skip this.
            
        return triples, triple_masks, used_pairs, current_alpha
    
    def csp_backtracking(target_size, time_limit=None):
        """
        Try to construct a system with independence number <= target_size
        using backtracking with forward checking.
        Returns (triples, triple_masks, used_pairs, alpha) if found, else None.
        """
        # We want to find a set of triples such that every subset of size target_size+1
        # contains at least one triple.
        # This is equivalent to: the complement of the independent set has size >= 16 - target_size.
        # Or: we need to "hit" all (target_size+1)-subsets.
        
        # Start with empty set of triples
        used_pairs = set()
        triples = []
        triple_masks = []
        
        # We'll use a greedy approach with backtracking
        # Key insight: if alpha > target_size, there's an independent set of size target_size+1
        # We need to add triples that hit this set.
        
        # For now, this is a placeholder. The main search is local search.
        return None
    
    best_triples = None
    best_triple_masks = None
    best_used_pairs = None
    best_indep = N + 1
    
    # Phase 1: Greedy constructions
    num_greedy = 300
    for i in range(num_greedy):
        if time.time() - start_time > DEADLINE:
            break
        
        triples, triple_masks, used_pairs = greedy_construct(i * 104729 + 13)
        if not triples:
            continue
        
        alpha, _ = compute_alpha_fast(triple_masks)
        
        if alpha < best_indep:
            best_indep = alpha
            best_triples = triples
            best_triple_masks = triple_masks
            best_used_pairs = used_pairs
        
        if best_indep <= 4:
            break
    
    # Phase 2: Local search on best solutions
    if best_triples is not None and time.time() - start_time < DEADLINE:
        time_limit_ls = min(DEADLINE - 5.0, time.time() - start_time + 80.0)
        improved_triples, improved_masks, improved_pairs, improved_alpha = local_search(
            best_triples, best_triple_masks, best_used_pairs, 
            max_iterations=100000, time_limit=time_limit_ls
        )
        if improved_alpha < best_indep:
            best_indep = improved_alpha
            best_triples = improved_triples
            best_triple_masks = improved_masks
            best_used_pairs = improved_pairs
    
    # Phase 3: More greedy + local search iterations
    for i in range(num_greedy, num_greedy + 500):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        triples, triple_masks, used_pairs = greedy_construct(i * 104729 + 13)
        if not triples:
            continue
        
        alpha, _ = compute_alpha_fast(triple_masks)
        
        if alpha < best_indep:
            best_indep = alpha
            best_triples = triples
            best_triple_masks = triple_masks
            best_used_pairs = used_pairs
            # Run local search on this promising solution
            if time.time() - start_time < DEADLINE - 15:
                time_limit_ls = min(DEADLINE - 10.0, time.time() - start_time + 30.0)
                improved_triples, improved_masks, improved_pairs, improved_alpha = local_search(
                    best_triples, best_triple_masks, best_used_pairs,
                    max_iterations=20000, time_limit=time_limit_ls
                )
                if improved_alpha < best_indep:
                    best_indep = improved_alpha
                    best_triples = improved_triples
                    best_triple_masks = improved_masks
                    best_used_pairs = improved_pairs
        
        if best_indep <= 4:
            break
    
    # Phase 4: Final polish on the best solution
    if best_triples is not None and time.time() - start_time < DEADLINE - 5:
        time_limit_final = DEADLINE - 3.0
        improved_triples, improved_masks, improved_pairs, improved_alpha = local_search(
            best_triples, best_triple_masks, best_used_pairs,
            max_iterations=100000, time_limit=time_limit_final
        )
        if improved_alpha < best_indep:
            best_indep = improved_alpha
            best_triples = improved_triples
            best_triple_masks = improved_masks
            best_used_pairs = improved_pairs
    
    if best_triples is None:
        # Fallback: return a simple valid system
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return [list(t) for t in best_triples]
