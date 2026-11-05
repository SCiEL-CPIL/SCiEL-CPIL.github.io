import time
import random
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = start + 55.0
    
    # All 4-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        m = 0
        for x in combo:
            m |= (1 << x)
        all_sets.append(m)
    
    def mask_to_list(m):
        return sorted((i+1) for i in range(16) if (m >> i) & 1)
    
    def can_add(C, F):
        """
        Checks if C can be added to F.
        Returns (True, None) if valid.
        Returns (False, (A, B)) if blocked by pair A, B.
        """
        for i in range(len(F)):
            for j in range(i+1, len(F)):
                v = F[i] & F[j]
                if (F[i] & C) == v and (F[j] & C) == v:
                    return False, (F[i], F[j])
        return True, None

    def greedy_pass(order):
        family = []
        for C in order:
            valid, _ = can_add(C, family)
            if valid:
                family.append(C)
        return family

    best_family = []
    best_size = 0
    
    # We will use a loop that alternates between greedy restarts and local search
    # to avoid getting stuck in a single local optimum.
    
    # Initial phase: Get a decent family
    # The "Inspiration 1" showed that deterministic/lexicographic order works well.
    # Let's try a few deterministic-ish orders and random ones.
    
    # Order 1: Natural (Lexicographic)
    fam1 = greedy_pass(all_sets)
    if len(fam1) > best_size:
        best_family = fam1
        best_size = len(fam1)
        
    # Order 2: Reverse Lexicographic
    fam2 = greedy_pass(all_sets[::-1])
    if len(fam2) > best_size:
        best_family = fam2
        best_size = len(fam2)
        
    # Order 3: Random
    for _ in range(5):
        if time.time() > DEADLINE: break
        order = all_sets[:]
        random.shuffle(order)
        fam3 = greedy_pass(order)
        if len(fam3) > best_size:
            best_family = fam3
            best_size = len(fam3)
            
    # Local Search Phase
    # We will run local search on the best family found so far.
    # If we have time, we might restart local search with a new greedy base.
    
    def local_search(family, deadline):
        family = list(family)
        in_family = set(family)
        candidates = [s for s in all_sets if s not in in_family]
        
        # Shuffle candidates to ensure variety
        random.shuffle(candidates)
        
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() < deadline:
            if not candidates:
                break
                
            # Pick a candidate
            # Instead of always index 0, pick random to avoid bias
            idx = random.randint(0, len(candidates) - 1)
            C = candidates[idx]
            
            valid, block_pair = can_add(C, family)
            
            if valid:
                # Add it
                family.append(C)
                in_family.add(C)
                candidates.pop(idx)
                no_improve_count = 0
                # Re-shuffle candidates occasionally to keep mixing
                if len(candidates) % 100 == 0:
                    random.shuffle(candidates)
            else:
                # Try to resolve the conflict
                # The conflict is caused by block_pair (A, B)
                # We can try to swap C with A or C with B
                A, B = block_pair
                
                improved = False
                
                # Try swapping A with C
                # Check if C is valid in family without A
                # We need to check C against all pairs in family \ {A}
                # This is O(N^2) again.
                
                # Optimization: We can just call can_add on a temporary list?
                # No, constructing a new list is O(N).
                # Let's write a helper that checks C against F excluding A.
                
                def can_add_except(C, F, excluded):
                    for i in range(len(F)):
                        if F[i] == excluded: continue
                        for j in range(i+1, len(F)):
                            if F[j] == excluded: continue
                            v = F[i] & F[j]
                            if (F[i] & C) == v and (F[j] & C) == v:
                                return False
                    return True

                # Try swap A
                if can_add_except(C, family, A):
                    # Perform swap
                    family.remove(A)
                    family.append(C)
                    in_family.discard(A)
                    in_family.add(C)
                    candidates.pop(idx)
                    candidates.append(A) # A becomes a candidate again
                    improved = True
                    no_improve_count = 0 # Configuration changed, might unlock others
                else:
                    # Try swap B
                    if can_add_except(C, family, B):
                        family.remove(B)
                        family.append(C)
                        in_family.discard(B)
                        in_family.add(C)
                        candidates.pop(idx)
                        candidates.append(B)
                        improved = True
                        no_improve_count = 0
                    else:
                        # Try a random swap with any member?
                        # This is more expensive. Let's limit this.
                        if no_improve_count % 10 == 0:
                            # Try random swap
                            if family:
                                rand_idx = random.randint(0, len(family)-1)
                                R = family[rand_idx]
                                if R != A and R != B:
                                    if can_add_except(C, family, R):
                                        family.remove(R)
                                        family.append(C)
                                        in_family.discard(R)
                                        in_family.add(C)
                                        candidates.pop(idx)
                                        candidates.append(R)
                                        improved = True
                                        no_improve_count = 0
                        if not improved:
                            no_improve_count += 1
                            # If stuck too long, restart candidates list
                            if no_improve_count > max_no_improve:
                                break
                                
        return family

    # Run local search on the best family
    if best_family:
        improved_family = local_search(best_family, DEADLINE - 2.0)
        if len(improved_family) > len(best_family):
            best_family = improved_family

    # If we still have time, do another round of greedy + local search
    if time.time() < DEADLINE - 5.0:
        for _ in range(3):
            if time.time() > DEADLINE - 5.0: break
            order = all_sets[:]
            random.shuffle(order)
            fam = greedy_pass(order)
            if len(fam) > len(best_family):
                best_family = fam
            
            # Try to improve this new family
            if time.time() < DEADLINE - 2.0:
                improved_fam = local_search(best_family, DEADLINE - 2.0)
                if len(improved_fam) > len(best_family):
                    best_family = improved_fam

    return [mask_to_list(m) for m in best_family]
