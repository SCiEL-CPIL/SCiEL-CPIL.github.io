# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 60.0  # Safety margin
    
    # Generate smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    # Precompute divisibility check for speed
    # divs[i][j] is True if smooth[i] divides smooth[j]
    # We only need to check pairs in the candidate set
    
    # Precompute set for O(1) lookup
    smooth_set = set(smooth)
    
    def is_antichain(terms):
        # terms is a sorted list of distinct smooth numbers
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                if terms[j] % terms[i] == 0:
                    return False
        return True

    def find_rep(n):
        # k=1
        if n in smooth_set:
            return [n]
        
        # k=2
        for i in range(len(smooth)):
            a = smooth[i]
            if a >= n:
                break
            b = n - a
            if b <= a:
                continue
            if b in smooth_set:
                if b % a != 0:
                    return [a, b]
        
        # k=3
        for i in range(len(smooth)):
            a = smooth[i]
            if a >= n:
                break
            for j in range(i + 1, len(smooth)):
                b = smooth[j]
                if a + b >= n:
                    break
                c = n - a - b
                if c <= b:
                    continue
                if c in smooth_set:
                    # a < b < c
                    if b % a != 0 and c % a != 0 and c % b != 0:
                        return [a, b, c]
        
        # k=4
        for i in range(len(smooth)):
            a = smooth[i]
            if a >= n:
                break
            for j in range(i + 1, len(smooth)):
                b = smooth[j]
                if a + b >= n:
                    break
                for k in range(j + 1, len(smooth)):
                    c = smooth[k]
                    if a + b + c >= n:
                        break
                    d = n - a - b - c
                    if d <= c:
                        continue
                    if d in smooth_set:
                        # a < b < c < d
                        if is_antichain([a, b, c, d]):
                            return [a, b, c, d]
        
        # Generic backtracking for k > 4
        # This should rarely be needed for n <= 128
        def backtrack(target, start_idx, current_sum, current_list):
            if current_sum == target:
                return current_list[:]
            if current_sum > target:
                return None
            # Pruning: if we have enough terms or start_idx out of bounds
            for idx in range(start_idx, len(smooth)):
                term = smooth[idx]
                if current_sum + term > target:
                    break
                # Check antichain property with new term
                valid = True
                for existing in current_list:
                    if term % existing == 0 or existing % term == 0:
                        # Since we pick from sorted list, term >= existing.
                        # If term % existing == 0, existing divides term. Invalid.
                        valid = False
                        break
                if valid:
                    current_list.append(term)
                    res = backtrack(target, idx + 1, current_sum + term, current_list)
                    if res:
                        return res
                    current_list.pop()
            return None

        res = backtrack(n, 0, 0, [])
        if res:
            return res
        
        # Absolute fallback: should not happen
        return [1] # 1 divides everything, so this is invalid. 
        # But n<=128 always has a solution. If we reach here, something is wrong.
        # To avoid crash, return a list.

    answers = []
    for n in range(1, 129):
        if time.time() > deadline - 5:
            # If time is tight, use a faster heuristic or precomputed?
            # For now, just proceed, as n<=128 is small.
            pass
        
        found = find_rep(n)
        # Ensure found is not None
        if found is None:
            # This should not happen, but handle it
            found = [n] if n in smooth_set else [1, n-1] # [1, n-1] might be invalid antichain if 1 divides n-1.
            # Let's just force a valid one if possible.
            # If find_rep failed, it's a bug. 
            # For safety, return a list to prevent crash.
            found = [1] 

        answers.append(found)
        
    return answers
# EVOLVE-BLOCK-END
