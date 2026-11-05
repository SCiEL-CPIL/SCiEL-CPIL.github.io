"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = start + 55.0  # Safety margin before 90s limit
    
    # All 4-subsets as bitmasks (16 bits, each bit represents element 0..15)
    all_sets = []
    for combo in combinations(range(16), 4):
        m = 0
        for x in combo:
            m |= (1 << x)
        all_sets.append(m)
    
    def mask_to_list(m):
        return sorted((i + 1) for i in range(16) if (m >> i) & 1)
    
    def is_valid_add(family, idx, C):
        """Check if C can be added without forming a sunflower triple."""
        for A in family:
            v = A & C
            if v in idx[A]:
                for B in idx[A][v]:
                    if (B & C) == v:
                        return False
        return True
    
    def add_set(family, idx, C):
        family.append(C)
        idx[C] = {}
        for A in family[:-1]:
            v = A & C
            if v not in idx[A]:
                idx[A][v] = []
            idx[A][v].append(C)
            if v not in idx[C]:
                idx[C][v] = []
            idx[C][v].append(A)
    
    def remove_set(family, idx, C):
        """Remove C from family and update indices."""
        family.remove(C)
        for A in family:
            v = A & C
            if v in idx[A] and C in idx[A][v]:
                idx[A][v].remove(C)
        if C in idx:
            del idx[C]
    
    def greedy_pass(order):
        family = []
        idx = {}
        for C in order:
            if is_valid_add(family, idx, C):
                add_set(family, idx, C)
        return family, idx
    
    def local_search(family, idx, all_sets, deadline):
        """Try to add more sets or swap to improve family size."""
        family = list(family)
        idx = {k: {kk: list(vv) for kk, vv in v.items()} for k, v in idx.items()}
        in_family = set(family)
        candidates = [s for s in all_sets if s not in in_family]
        random.shuffle(candidates)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for s in candidates:
                if time.time() >= deadline:
                    break
                if s in in_family:
                    continue
                if is_valid_add(family, idx, s):
                    add_set(family, idx, s)
                    in_family.add(s)
                    improved = True
                    break
                # Try swap: remove one member, add s
                for rem_idx in range(min(5, len(family))):
                    if time.time() >= deadline:
                        break
                    old = family[rem_idx]
                    remove_set(family, idx, old)
                    in_family.discard(old)
                    if is_valid_add(family, idx, s):
                        add_set(family, idx, s)
                        in_family.add(s)
                        improved = True
                        break
                    else:
                        add_set(family, idx, old)
                        in_family.add(old)
                if improved:
                    break
            in_family = set(family)
            candidates = [s for s in all_sets if s not in in_family]
            random.shuffle(candidates)
        
        return family, idx
    
    def local_search_v2(family, deadline):
        """
        Structural change: Use pairwise conflict resolution.
        For each candidate C, if blocked by a pair (A,B), try swapping C with A or B.
        This is a different search neighborhood than the current program's "remove any 1 member".
        """
        family = list(family)
        in_family = set(family)
        candidates = [s for s in all_sets if s not in in_family]
        random.shuffle(candidates)
        
        def can_add_except(C, F, excluded):
            """Check if C is valid in F excluding 'excluded'."""
            for i in range(len(F)):
                if F[i] == excluded: continue
                for j in range(i+1, len(F)):
                    if F[j] == excluded: continue
                    v = F[i] & F[j]
                    if (F[i] & C) == v and (F[j] & C) == v:
                        return False
            return True

        def get_blocker(C, F):
            """Find a blocking pair for C in F, or None if valid."""
            for i in range(len(F)):
                for j in range(i+1, len(F)):
                    v = F[i] & F[j]
                    if (F[i] & C) == v and (F[j] & C) == v:
                        return (F[i], F[j])
            return None

        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline:
            if not candidates:
                break
            
            # Pick a random candidate
            idx = random.randint(0, len(candidates) - 1)
            C = candidates[idx]
            
            if C in in_family:
                candidates.pop(idx)
                continue
                
            blocker = get_blocker(C, family)
            
            if blocker is None:
                # Valid, add it
                family.append(C)
                in_family.add(C)
                candidates.pop(idx)
                no_improve = 0
            else:
                A, B = blocker
                improved = False
                
                # Try swapping A
                if can_add_except(C, family, A):
                    family.remove(A)
                    family.append(C)
                    in_family.discard(A)
                    in_family.add(C)
                    candidates.pop(idx)
                    candidates.append(A)
                    improved = True
                    no_improve = 0
                else:
                    # Try swapping B
                    if can_add_except(C, family, B):
                        family.remove(B)
                        family.append(C)
                        in_family.discard(B)
                        in_family.add(C)
                        candidates.pop(idx)
                        candidates.append(B)
                        improved = True
                        no_improve = 0
                    else:
                        # Try random swap
                        if no_improve % 10 == 0 and family:
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
                                    no_improve = 0
                        
                        if not improved:
                            no_improve += 1
                            if no_improve > max_no_improve:
                                break
                            # Shuffle candidates to break bias
                            if no_improve % 100 == 0:
                                random.shuffle(candidates)
                            
        return family

    best_family = []
    random.seed(42)
    
    # Phase 1: Greedy restarts with the fast indexed method
    # We run multiple passes, keeping the best
    # The current program does this well.
    
    num_passes = 0
    while time.time() < DEADLINE - 10.0:  # Leave 10s for local search
        num_passes += 1
        order = all_sets[:]
        random.shuffle(order)
        fam, idx = greedy_pass(order)
        
        if len(fam) > len(best_family):
            best_family = fam
        
        if len(best_family) >= 250:
            break
    
    # Phase 2: Local search
    # Try both the old local search and the new structural one.
    
    if best_family and time.time() < DEADLINE - 5.0:
        # Rebuild idx for best_family
        idx = {}
        for i, C in enumerate(best_family):
            idx[C] = {}
            for A in best_family[:i]:
                v = A & C
                if v not in idx[A]:
                    idx[A][v] = []
                idx[A][v].append(C)
                if v not in idx[C]:
                    idx[C][v] = []
                idx[C][v].append(A)
        
        improved_fam, _ = local_search(best_family, idx, all_sets, DEADLINE - 8.0)
        if len(improved_fam) > len(best_family):
            best_family = improved_fam
            
    if best_family and time.time() < DEADLINE - 2.0:
        improved_fam_v2 = local_search_v2(best_family, DEADLINE - 1.0)
        if len(improved_fam_v2) > len(best_family):
            best_family = improved_fam_v2

    return [mask_to_list(m) for m in best_family]
# EVOLVE-BLOCK-END
