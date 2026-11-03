import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    # The task specifies q=3, n=8, radius=1.
    N = q ** n
    powers = [q ** i for i in range(n)]
    
    # Helper to decode an integer to a list of digits
    def decode(x):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits

    # Precompute coverage bitsets using numpy for speed
    # coverage[c] is an integer where bit i is 1 if point i is covered by codeword c
    coverage = [0] * N
    
    # Precompute neighbor masks for each point
    for c in range(N):
        bits = 1 << c
        for i in range(n):
            base = powers[i]
            val = (c // base) % q
            for new_val in range(q):
                if new_val == val:
                    continue
                neighbor = c + (new_val - val) * base
                bits |= (1 << neighbor)
        coverage[c] = bits

    full_mask = (1 << N) - 1
    
    # Tetracode product construction for structured initialization
    # The tetracode is a perfect code of length 4 over Z_3 with 9 codewords
    # covering radius 1. We can use it as a building block.
    # Tetracode codewords (known optimal covering code K_3(4,1) = 9):
    tetracode = [
        [0,0,0,0], [0,1,2,1], [0,2,1,2],
        [1,0,1,2], [1,1,0,1], [1,2,2,0],
        [2,0,2,1], [2,1,1,0], [2,2,0,2]
    ]
    
    # Verify tetracode covers all of Z_3^4
    # For each position i and value v, check coverage
    def tetracode_cover():
        """Return set of tetracode codeword indices that cover each point"""
        cover_map = {}
        for idx, tc in enumerate(tetracode):
            for i in range(4):
                for v in range(3):
                    # Point at position i with value v is covered by tc if
                    # tc[i] == v or tc differs from point only at position i
                    # Actually, we need: point p is within distance 1 of tc
                    # p is within dist 1 of tc iff they agree in at least 3 positions
                    pass
        return cover_map
    
    # Better: build the tetracode-based initialization directly
    # For n=8, split into two halves of 4.
    # A codeword (a,b) where a is a tetracode word (first 4) and b is a tetracode word (last 4)
    # This gives 9*9 = 81 codewords. Each covers points within dist 1 in each half.
    # But this isn't quite right for the full space.
    
    # Actually, let's use a different structured approach.
    # The known best construction uses the tetracode in a more sophisticated way.
    # Let's use greedy with smart ordering as fallback, plus local search.
    
    best_code = None
    best_size = N + 1
    
    # Strategy 1: Greedy Set Cover with structured ordering
    # Order candidates by their "efficiency" - points they uniquely cover
    
    num_iterations = 15
    
    for iteration in range(num_iterations):
        if time.time() > deadline - 15:
            break
            
        uncovered = full_mask
        selected = []
        
        # Precompute a random perturbation for diversity
        rng = random.Random(iteration * 7919 + 42)
        
        # Use a priority queue approach: repeatedly pick the codeword that covers
        # the most uncovered points, with random tie-breaking
        while uncovered:
            best_c = -1
            best_gain = 0
            # Sample a subset of candidates for speed, but ensure we check all
            # if N is small enough
            candidates = list(range(N))
            rng.shuffle(candidates)
            
            for c in candidates:
                gain = (uncovered & coverage[c]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                elif gain == best_gain and best_gain > 0:
                    if rng.random() < 0.3:
                        best_c = c
            
            if best_gain == 0:
                break
                
            selected.append(best_c)
            uncovered &= ~coverage[best_c]
            
        if uncovered != 0:
            continue
        
        # Local Search: Redundancy Removal with count-based checking
        # Build count array: count[p] = number of selected codewords covering point p
        count = np.zeros(N, dtype=np.int32)
        for c in selected:
            # Add coverage of c to count
            mask = coverage[c]
            # Extract set bits and increment
            m = mask
            while m:
                # Get lowest set bit
                lb = m & (-m)
                idx = lb.bit_length() - 1
                count[idx] += 1
                m ^= lb
        
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            indices = list(range(len(selected)))
            rng.shuffle(indices)
            for i in indices:
                if i >= len(selected):
                    continue
                c = selected[i]
                
                # Check if any point is covered only by c
                # i.e., count[p] == 1 for some p in coverage[c]
                mask = coverage[c]
                m = mask
                redundant = True
                while m:
                    lb = m & (-m)
                    idx = lb.bit_length() - 1
                    if count[idx] == 1:
                        redundant = False
                        break
                    m ^= lb
                
                if redundant:
                    # Remove c
                    # Decrement counts
                    m = coverage[c]
                    while m:
                        lb = m & (-m)
                        idx = lb.bit_length() - 1
                        count[idx] -= 1
                        m ^= lb
                    selected.pop(i)
                    changed = True
                    break  # Restart after modification
        
        # Local Search: Swap optimization
        # Try replacing a codeword with another that allows more removals
        if time.time() < deadline - 5:
            swap_improved = True
            while swap_improved and time.time() < deadline - 3:
                swap_improved = False
                for i in range(len(selected)):
                    if time.time() > deadline - 3:
                        break
                    c = selected[i]
                    # Try removing c and replacing with a better candidate
                    # First, check what would be uncovered if we remove c
                    # Points where count == 1 and in coverage[c]
                    mask = coverage[c]
                    m = mask
                    only_covered = []
                    while m:
                        lb = m & (-m)
                        idx = lb.bit_length() - 1
                        if count[idx] == 1:
                            only_covered.append(idx)
                        m ^= lb
                    
                    if not only_covered:
                        # c is redundant, already handled
                        continue
                    
                    # Find a candidate d that covers all only_covered points
                    # and potentially allows further reduction
                    best_d = -1
                    best_score = len(only_covered)  # minimum score to beat
                    for d in range(N):
                        if d == c:
                            continue
                        # Check if d covers all only_covered points
                        covers_all = True
                        for p in only_covered:
                            if not (coverage[d] >> p) & 1:
                                covers_all = False
                                break
                        if not covers_all:
                            continue
                        # Score: how many additional points does d cover that are
                        # currently covered by only 1 other codeword (besides c)?
                        # This is complex; use a simpler heuristic
                        # Count how many points in coverage[d] have count <= 2
                        # (excluding c's contribution)
                        m2 = coverage[d]
                        score = 0
                        while m2:
                            lb = m2 & (-m2)
                            idx = lb.bit_length() - 1
                            if count[idx] <= 2:
                                score += 1
                            m2 ^= lb
                        if score > best_score:
                            best_score = score
                            best_d = d
                    
                    if best_d != -1:
                        # Perform swap: remove c, add best_d
                        # Update counts
                        m = coverage[c]
                        while m:
                            lb = m & (-m)
                            idx = lb.bit_length() - 1
                            count[idx] -= 1
                            m ^= lb
                        selected[i] = best_d
                        m = coverage[best_d]
                        while m:
                            lb = m & (-m)
                            idx = lb.bit_length() - 1
                            count[idx] += 1
                            m ^= lb
                        swap_improved = True
                        break
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = selected[:]
            
            # Continue local search on the improved solution
            if time.time() < deadline - 5:
                # Rebuild count for the new best
                count2 = np.zeros(N, dtype=np.int32)
                for c in best_code:
                    m = coverage[c]
                    while m:
                        lb = m & (-m)
                        idx = lb.bit_length() - 1
                        count2[idx] += 1
                        m ^= lb
                
                changed2 = True
                while changed2 and time.time() < deadline - 3:
                    changed2 = False
                    indices2 = list(range(len(best_code)))
                    rng.shuffle(indices2)
                    for i in indices2:
                        if i >= len(best_code):
                            continue
                        c = best_code[i]
                        mask = coverage[c]
                        m = mask
                        redundant = True
                        while m:
                            lb = m & (-m)
                            idx = lb.bit_length() - 1
                            if count2[idx] == 1:
                                redundant = False
                                break
                            m ^= lb
                        if redundant:
                            m = coverage[c]
                            while m:
                                lb = m & (-m)
                                idx = lb.bit_length() - 1
                                count2[idx] -= 1
                                m ^= lb
                            best_code.pop(i)
                            changed2 = True
                            break
    
    # If we still have time, do a final polish
    if best_code is not None and time.time() < deadline - 2:
        count_final = np.zeros(N, dtype=np.int32)
        for c in best_code:
            m = coverage[c]
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                count_final[idx] += 1
                m ^= lb
        
        changed_final = True
        while changed_final and time.time() < deadline - 1:
            changed_final = False
            indices_final = list(range(len(best_code)))
            random.shuffle(indices_final)
            for i in indices_final:
                if i >= len(best_code):
                    continue
                c = best_code[i]
                mask = coverage[c]
                m = mask
                redundant = True
                while m:
                    lb = m & (-m)
                    idx = lb.bit_length() - 1
                    if count_final[idx] == 1:
                        redundant = False
                        break
                    m ^= lb
                if redundant:
                    m = coverage[c]
                    while m:
                        lb = m & (-m)
                        idx = lb.bit_length() - 1
                        count_final[idx] -= 1
                        m ^= lb
                    best_code.pop(i)
                    changed_final = True
                    break
    
    # Fallback
    if best_code is None:
        best_code = list(range(N))
    
    result = []
    for c in best_code:
        result.append(decode(c))
        
    return result
