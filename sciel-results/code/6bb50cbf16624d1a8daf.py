import time
import itertools
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if n == 1:
        return "1"
    
    if n == 2:
        return "121"

    # 1. Generate standard superpermutation
    # The standard recursive construction produces a word of length (n-1)! * n + n - 1 ? No.
    # Length L_n = n * L_{n-1} - (n-1)
    # L_1 = 1
    # L_2 = 2*1 - 1 = 2? No, "12" is not a superpermutation. "121" is.
    # Let's check the standard construction again.
    # S_1 = "1"
    # S_2 = "121" (Len 3)
    # S_3 = "1213121" (Len 7)
    # S_4 = "121312141213121" (Len 13)
    # Formula: L_n = n! + (n-1)! + (n-2)! + ... + 1! - (n-1) ?
    # Actually, the length of the standard superpermutation is:
    # L_n = n * L_{n-1} - (n-1)
    # L_1 = 1
    # L_2 = 2*1 - 1 = 2? But "12" doesn't contain "21".
    # The standard construction often cited is:
    # S_n = S_{n-1} n S_{n-1} n ... (interleaved)
    # Actually, the "standard" superpermutation of length 5905 is NOT the simple recursive one.
    # The simple recursive one has length:
    # n=1: 1
    # n=2: 3 ("121")
    # n=3: 9 ("121312131"?) No.
    # Let's generate the standard one carefully.
    # S_1 = [1]
    # S_2 = [1, 2, 1]
    # S_3 = [1, 2, 1, 3, 1, 2, 1, 3, 1]? No.
    
    # The standard construction is:
    # S_n = S_{n-1} n S_{n-1} n ... (repeated n times) with overlaps?
    # Actually, the known "standard" superpermutation for n=7 has length 5905.
    # The simple recursive formula L_n = n * L_{n-1} - (n-1) gives:
    # L_1 = 1
    # L_2 = 2*1 - 1 = 2 (Invalid, must be at least 3)
    # So the base case is usually handled specially or the formula is L_n = n * L_{n-1} - (n-1) for n>=3?
    # L_2 = 3
    # L_3 = 3*3 - 2 = 7
    # L_4 = 4*7 - 3 = 25
    # L_5 = 5*25 - 4 = 121
    # L_6 = 6*121 - 5 = 721
    # L_7 = 7*721 - 6 = 5041
    
    # Wait, the task says "The shortest known word has 5905 symbols".
    # 5041 is shorter than 5905.
    # Is 5041 a valid superpermutation?
    # Yes, the recursive construction S_n = S_{n-1} n S_{n-1} n ... is valid.
    # Let's verify for n=3.
    # S_2 = "121"
    # S_3 = "121" "3" "121" "3" "121" ?
    # "12131213121"
    # Length 11.
    # Formula 3*3 - 2 = 7.
    # "1213121"
    # Windows of 3: 121, 213, 131, 312, 121.
    # Perms of 1,2,3: 123, 132, 213, 231, 312, 321.
    # Missing 123, 132, 231, 321.
    # So "1213121" is NOT a superpermutation.
    
    # The correct recursive construction is:
    # S_n = S_{n-1} n S_{n-1} n ... (n times)
    # S_1 = "1"
    # S_2 = "121"
    # S_3 = "121" "3" "121" "3" "121" -> "12131213121" (Length 11)
    # Windows:
    # 121, 213, 131, 312, 121, 213, 131, 312, 121
    # Still missing many.
    
    # Actually, the standard construction is:
    # S_n = S_{n-1} n S_{n-1} n ... (n times) is WRONG.
    
    # The correct one is:
    # S_n = S_{n-1} n S_{n-1} n ... (n times) where S_{n-1} is the superpermutation of n-1.
    # But we need to interleave them properly.
    
    # Let's look at the known length 5905.
    # The standard recursive construction yields:
    # L_n = n! + (n-1)! + (n-2)! + ... + 1! - (n-1)
    # For n=7:
    # 5040 + 720 + 120 + 24 + 6 + 2 + 1 - 6 = 5907?
    # 5040 + 720 = 5760
    # 5760 + 120 = 5880
    # 5880 + 24 = 5904
    # 5904 + 6 = 5910
    # 5910 + 2 = 5912
    # 5912 + 1 = 5913
    # 5913 - 6 = 5907.
    
    # The task says 5905.
    # Maybe the formula is slightly different or there's a better construction.
    # However, the seed program uses a specific construction.
    # Let's stick to the seed program's construction logic which produced a valid result (score 1.0).
    # The seed program's construction:
    # word = [1]
    # for m in range(2, n+1):
    #   ...
    #   block = list(p) + [m] + list(p)
    #   ...
    
    # This construction is:
    # S_m = concat( p + [m] + p for p in perms(S_{m-1}) ) with overlaps.
    
    # Let's implement this construction correctly and quickly.
    
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
            
        # Extract unique (m-1)-permutations from word
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build new word
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap with current new_word
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        
        word = new_word
    
    S = list(word)
    
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Initialize counts
    perm_counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i + n])
        perm_counts[p] = perm_counts.get(p, 0) + 1
        
    # Pruning Loop
    # The seed program's pruning loop is O(N^2) in the worst case per pass, and it does 100 passes.
    # For N=5905, this is very slow in Python.
    # We need a faster pruning strategy or accept the seed's result if it's too slow.
    # However, the seed scored 1.0, meaning it finished in time.
    # Let's optimize the pruning check.
    
    # Optimization:
    # Instead of checking every index i from 0 to N-1 in every pass,
    # we can maintain a list of "candidates" for removal.
    # An index i is a candidate if the window starting at i-n+1, i-n+2, ..., i
    # has counts that allow removal.
    
    # Actually, the seed's `is_valid_incremental` is O(n) per check.
    # And it checks N indices per pass.
    # N=5905, n=7.
    # 5905 * 7 = ~40k operations per pass.
    # 100 passes = 4M operations.
    # This should be fast enough in Python if optimized.
    
    def is_valid_incremental(S, i, perm_counts):
        L = len(S)
        if L <= n:
            return False, None, None
        
        start_min = max(0, i - n + 1)
        start_max = min(i, len(S) - n)
        
        deltas = {}
        
        # Old windows to remove
        for j in range(start_min, start_max + 1):
            w = tuple(S[j:j+n])
            deltas[w] = deltas.get(w, 0) - 1
            
        # New windows to add
        for j in range(max(0, i - n + 1), i):
            if j > len(S) - 1 - n:
                continue
            part1 = S[j:i]
            part2 = S[i+1:j+n+1]
            new_win = tuple(part1 + part2)
            deltas[new_win] = deltas.get(new_win, 0) + 1
            
        for p, delta in deltas.items():
            if p in all_perms:
                new_count = perm_counts.get(p, 0) + delta
                if new_count < 1:
                    return False, None, None
                    
        return True, S[:i] + S[i+1:], deltas

    # Pruning
    for _pass in range(100):
        if time.time() > deadline - 5:
            break
        removed = False
        # Iterate backwards to avoid index shifting issues within the pass
        # But we break after first removal, so it's safe.
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            if len(S) <= n:
                break
            valid, new_S, deltas = is_valid_incremental(S, i, perm_counts)
            if valid:
                S = new_S
                for p, delta in deltas.items():
                    perm_counts[p] = perm_counts.get(p, 0) + delta
                    if perm_counts[p] == 0:
                        del perm_counts[p]
                removed = True
                break
        if not removed:
            break
            
    return "".join(map(str, S))

# Test locally if needed
# if __name__ == "__main__":
#     print(construct(3))
#     print(construct(4))
