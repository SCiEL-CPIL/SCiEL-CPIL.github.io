import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def calculate_independence_number(system):
        """
        Efficiently calculates the independence number of a linear 3-uniform hypergraph on 16 points.
        Uses bitmask representation for speed.
        """
        if not system:
            return 16
            
        triple_masks = []
        for a, b, c in system:
            mask = (1 << a) | (1 << b) | (1 << c)
            triple_masks.append(mask)
        
        # We want the largest set S such that no triple is a subset of S.
        # Equivalent to: S does not contain any triple.
        # We can iterate by size from 16 down to 1.
        # But 2^16 = 65536 iterations is very fast in Python if optimized.
        
        max_size = 0
        # Precompute popcount for all 16-bit masks? No, bin(mask).count('1') is fast enough for 65k calls.
        # Or use a lookup table.
        popcount = [0] * (1 << 16)
        for i in range(1, 1 << 16):
            popcount[i] = popcount[i >> 1] + (i & 1)
            
        # Optimization: If we find an independent set of size k, we don't need to check smaller ones.
        # We iterate masks in descending order of popcount? 
        # Generating masks in descending popcount order is complex.
        # Instead, we can just iterate all and track max.
        
        # A better heuristic for small n: 
        # Check all subsets of size 16, 15, ... until we find one that is independent.
        # Since we want to MINIMIZE the independence number, we expect it to be small (e.g., 5 or 6).
        # So we can check if there exists an independent set of size 7. If not, check 6, etc.
        
        # Let's check from high to low.
        # To do this efficiently, we can generate combinations of points.
        # But checking "contains no triple" is O(E).
        
        # Let's stick to the bitmask iteration but break early if possible?
        # Actually, just finding the MAX is required.
        
        best_alpha = 0
        for mask in range(1 << 16):
            is_independent = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    is_independent = False
                    break
            if is_independent:
                size = popcount[mask]
                if size > best_alpha:
                    best_alpha = size
                    # If we hit 16, it's impossible unless system is empty.
                    # If we hit a high number, we might want to stop? No, we need the max.
        
        return best_alpha

    def is_valid_system(triples):
        """Check linearity: no pair appears in more than one triple."""
        seen_pairs = set()
        for t in triples:
            a, b, c = t
            pairs = [(a, b), (a, c), (b, c)]
            for p in pairs:
                p_sorted = tuple(sorted(p))
                if p_sorted in seen_pairs:
                    return False
                seen_pairs.add(p_sorted)
        return True

    def greedy_packing(seed):
        """
        Construct a maximal linear triple system using a greedy approach with randomization.
        We try to pack as many triples as possible (max 40).
        """
        random.seed(seed)
        used_pairs = set()
        triples = []
        
        # Generate all possible triples
        all_triples = list(combinations(range(16), 3))
        random.shuffle(all_triples)
        
        for t in all_triples:
            a, b, c = t
            p1 = (a, b) if a < b else (b, a)
            p2 = (a, c) if a < c else (c, a)
            p3 = (b, c) if b < c else (c, b)
            
            if p1 not in used_pairs and p2 not in used_pairs and p3 not in used_pairs:
                triples.append([a, b, c])
                used_pairs.add(p1)
                used_pairs.add(p2)
                used_pairs.add(p3)
                
                # Max possible is 40. If we reach 40, we can stop.
                if len(triples) == 40:
                    break
                    
        return triples, used_pairs

    def local_search(triples, used_pairs, time_limit):
        """
        Try to improve the system by swapping triples.
        A swap removes one triple and adds another.
        To keep linearity, the new triple must only use pairs that are:
        1. Freed by the removal of the old triple.
        2. Not used by any other triple.
        
        However, a simple 1-for-1 swap is very restrictive.
        If we remove T1={a,b,c}, we free pairs {ab, ac, bc}.
        We can only add T2 if its pairs are a subset of these freed pairs?
        No, T2 has 3 pairs. If T2 uses a pair not in {ab, ac, bc}, that pair is still used by some other triple (since the system was maximal or near-maximal).
        Wait, if the system is not maximal, there might be free pairs.
        
        Let's check for "holes".
        Actually, a more powerful move is a "cycle swap" or just random re-insertion.
        
        Let's try a different local search: 
        1. Identify an independent set S of size alpha.
        2. Try to "break" S by adding a triple that intersects S in 2 or 3 points? 
           No, we can't add a triple that intersects an existing triple in 2 points.
           
        Let's try a 2-opt style swap:
        Remove T1 and T2. Free pairs P1 and P2.
        Add T3 and T4 such that their pairs are subsets of P1 U P2 and don't conflict.
        
        This is complex. Let's try a simpler heuristic:
        Randomly pick a triple to remove.
        Randomly pick a new triple from the "available" pool (triples that could fit if we ignored the removed one? No, just check validity against the rest).
        
        Algorithm:
        1. Start with greedy system.
        2. Calculate alpha.
        3. While time remains:
           a. Pick a random triple T_remove.
           b. Remove it. Update used_pairs.
           c. Generate a candidate T_add. 
              - Ideally, T_add should cover pairs that were in T_remove to maintain density.
              - Or just pick a random valid triple from the remaining space.
           d. If T_add is valid (pairs not in used_pairs), accept.
           e. Calculate new alpha.
           f. If alpha improved or (same alpha and random probability), accept. Else rollback.
        """
        current_triples = triples
        current_pairs = used_pairs
        current_alpha = calculate_independence_number(current_triples)
        
        best_triples = current_triples
        best_alpha = current_alpha
        
        # Precompute all valid triples for faster lookup? 
        # There are 560 triples.
        
        all_triples_list = list(combinations(range(16), 3))
        
        # For speed, let's just try random swaps.
        
        while time.time() < time_limit:
            # Pick a triple to remove
            if not current_triples:
                break
            idx_remove = random.randint(0, len(current_triples) - 1)
            t_remove = current_triples[idx_remove]
            
            # Remove it
            a, b, c = t_remove
            p1 = (a, b) if a < b else (b, a)
            p2 = (a, c) if a < c else (c, a)
            p3 = (b, c) if b < c else (c, b)
            
            current_pairs.discard(p1)
            current_pairs.discard(p2)
            current_pairs.discard(p3)
            current_triples.pop(idx_remove)
            
            # Try to find a valid triple to add
            # We prefer triples that use the freed pairs to maintain density
            # Let's try a few candidates
            found_add = False
            new_alpha = current_alpha + 1 # Optimistic
            
            # Try candidates that share pairs with t_remove first
            candidates = []
            for t in all_triples_list:
                # Check if t shares at least one pair with t_remove
                ta, tb, tc = t
                pairs_t = [(ta, tb), (ta, tc), (tb, tc)]
                pairs_t_sorted = [tuple(sorted(p)) for p in pairs_t]
                
                shares = False
                for pt in pairs_t_sorted:
                    if pt in (p1, p2, p3):
                        shares = True
                        break
                if shares:
                    candidates.append(t)
            
            # Also add some random candidates
            for _ in range(5):
                candidates.append(random.choice(all_triples_list))
            
            random.shuffle(candidates)
            
            for t_add in candidates:
                if time.time() > time_limit:
                    break
                ta, tb, tc = t_add
                pa = (ta, tb) if ta < tb else (tb, ta)
                pb = (ta, tc) if ta < tc else (tc, ta)
                pc = (tb, tc) if tb < tc else (tc, tb)
                
                if pa not in current_pairs and pb not in current_pairs and pc not in current_pairs:
                    # Valid addition
                    current_triples.append(t_add)
                    current_pairs.add(pa)
                    current_pairs.add(pb)
                    current_pairs.add(pc)
                    found_add = True
                    
                    # Calculate alpha
                    # Optimization: only calculate if we think it might be better?
                    # We need to know if it's better.
                    new_alpha = calculate_independence_number(current_triples)
                    
                    if new_alpha < current_alpha:
                        # Improvement! Accept and update best
                        current_alpha = new_alpha
                        if new_alpha < best_alpha:
                            best_alpha = new_alpha
                            best_triples = [list(t) for t in current_triples]
                        break
                    elif new_alpha == current_alpha:
                        # Neutral move. Accept with some probability to allow exploration?
                        # Or just accept to change the landscape.
                        if random.random() < 0.5:
                            # Keep it, but don't update best unless we find something better later
                            # Actually, if we keep it, current_alpha stays same.
                            break
                        else:
                            # Rollback
                            current_triples.pop()
                            current_pairs.discard(pa)
                            current_pairs.discard(pb)
                            current_pairs.discard(pc)
                            found_add = False
                            continue
                    else:
                        # Worse. Rollback.
                        current_triples.pop()
                        current_pairs.discard(pa)
                        current_pairs.discard(pb)
                        current_pairs.discard(pc)
                        found_add = False
                        continue
            
            if not found_add:
                # Rollback removal
                current_triples.insert(idx_remove, t_remove)
                current_pairs.add(p1)
                current_pairs.add(p2)
                current_pairs.add(p3)
                
                # If we couldn't add anything, the system is maximal in this local view.
                # Try a different removal strategy or break?
                # Let's just continue trying other removals.
                
            if best_alpha <= 5:
                break

        return best_triples, best_alpha

    # Main Logic
    best_system = None
    best_score = 16
    
    # 1. Try a few different greedy seeds to find a good starting point
    # We want high density (many triples) to minimize independence number.
    
    # Run greedy with different seeds
    for i in range(50):
        if time.time() > deadline - 10:
            break
        triples, used_pairs = greedy_packing(i)
        alpha = calculate_independence_number(triples)
        
        if alpha < best_score:
            best_score = alpha
            best_system = triples
            if best_score <= 5:
                break
        
        # If we have a decent system (e.g., alpha <= 7), run local search on it
        if alpha <= 8 and time.time() < deadline - 20:
            # Allocate time for local search
            # If we have 100s left, use 10s for this local search?
            # Let's use a fixed budget per local search or proportional.
            ls_time_limit = time.time() + min(10.0, deadline - time.time() - 5.0)
            if ls_time_limit > time.time():
                improved_triples, improved_alpha = local_search(triples, used_pairs, ls_time_limit)
                if improved_alpha < best_score:
                    best_score = improved_alpha
                    best_system = improved_triples
                    if best_score <= 5:
                        break

    # If we still have time, try to refine the best system found
    if best_system and best_score > 5 and time.time() < deadline - 5:
        # Reconstruct used_pairs for best_system
        used_pairs = set()
        for t in best_system:
            a, b, c = t
            used_pairs.add((a, b) if a < b else (b, a))
            used_pairs.add((a, c) if a < c else (c, a))
            used_pairs.add((b, c) if b < c else (c, b))
        
        final_time_limit = deadline - 2.0
        final_triples, final_alpha = local_search(best_system, used_pairs, final_time_limit)
        if final_alpha < best_score:
            best_score = final_alpha
            best_system = final_triples

    return best_system

