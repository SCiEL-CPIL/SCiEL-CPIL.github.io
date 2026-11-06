import time
import random

N = 256

def is_sidon(S):
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def is_maximal(S):
    if not is_sidon(S):
        return False
    S_set = set(S)
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
            
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if x can be added
        valid = True
        # Check diagonal
        if 2 * x in sums:
            valid = False
        else:
            # Check off-diagonal
            for s in S:
                if x + s in sums:
                    valid = False
                    break
        if valid:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 20.0  # Safety margin
    
    # 1. Try specific small candidates
    candidates = []
    
    # Powers of 2
    pow2 = [1, 2, 4, 8, 16, 32, 64, 128]
    if is_maximal(pow2):
        return pow2
        
    # Try some other small structures?
    # Randomized search for minimal maximal set
    
    best_set = None
    best_size = float('inf')
    
    # Seed random for reproducibility? No, let it vary.
    # But for stability in evaluation, maybe seed with 42?
    # The prompt says "seed/value" where seed is the score of the seed program.
    # It doesn't imply the random seed.
    
    while time.time() < deadline:
        # Random permutation
        perm = list(range(1, N + 1))
        random.shuffle(perm)
        
        S = []
        sums = set()
        
        for x in perm:
            # Check if x can be added to S
            added = [x + a for a in S] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                S.append(x)
                sums.update(added)
        
        # The resulting S is maximal by construction (greedy saturation)
        # Check if it's actually Sidon (should be) and Maximal
        # Greedy saturation ensures maximality with respect to the *order*?
        # No, greedy saturation ensures that no element *later* in the permutation could be added.
        # But what about elements *earlier* that were skipped?
        # If an element was skipped, it was because it conflicted with the current S at that time.
        # Since S only grows, any conflict that existed then still exists now.
        # So no skipped element can be added.
        # So the set is maximal.
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            
            # If we find a very small set, we can stop early?
            # Theoretical lower bound is ~ log N?
            # For N=256, size 6 or 7 might be possible.
            if best_size <= 6:
                break

    if best_set is not None:
        return best_set
    else:
        # Fallback: return the seed program's result
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected

# EVOLVE-BLOCK-END
