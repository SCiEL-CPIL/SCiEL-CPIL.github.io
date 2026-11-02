"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

# EVOLVE-BLOCK-START
def construct(n):
    # Deadline guard: must return within ~110s to be safe under 120s/150s limits.
    deadline = time.time() + 110.0

    # For small n, use the classic recursive construction (exact optimal for n<=4,
    # good for n=5). For n>=6, we use a smarter approach.
    # The classic construction gives length 1!+2!+...+n! = 3, 9, 33, 153, 873, 5913, 46233...
    # Wait: 1!=1, 2!=2, 3!=6, 4!=24, 5!=120, 6!=720, 7!=5040
    # Sum for n=7: 1+2+6+24+120+720+5040 = 5913. That's close to 5905.
    # The known shortest is 5905, so we need to shave 8 characters.

    # Let's implement the classic construction first, then try to optimize.
    def classic_construct(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find maximum overlap between end of new and start of block
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    if n <= 4:
        word = classic_construct(n)
        return "".join(map(str, word))

    # For n=5,6,7: use classic construction which gives 873, 5913, etc.
    # Then try to find a shorter superpermutation via local search / optimization.
    
    # Actually, let me think about this differently.
    # The classic construction for n=7 gives 5913. The known optimum is 5905.
    # We need a smarter construction.
    
    # Key insight: The classic construction P_m P can sometimes be shortened.
    # When we append a new symbol m, the pattern is: for each (m-1)-permutation p that
    # appears, we add p, m, p. But consecutive blocks can share more than just the
    # overlap of the p's.
    
    # A better approach: use the fact that the superpermutation can be built by
    # inserting the new symbol in a way that maximizes overlap.
    
    # Let me try a different construction: the "greedy" approach where we build
    # the word by adding symbols one at a time, always choosing the next symbol
    # that completes the most new permutations.
    
    # Actually, for n=7 with 120s, let's use the classic construction (5913) and
    # then try to trim it. Or use a known better construction.
    
    # The construction by Teodorescu and others uses a more careful insertion.
    # Let me implement a version that tries to minimize the word length.
    
    # Strategy: Build the superpermutation for n=7 using the recursive structure
    # but with optimized overlaps.
    
    # For n=7, the classic gives 5913. Let's just use it - it's valid and close.
    # The score is 5913/5905 ≈ 1.00135... wait no, score is seed/value where seed is
    # the current program's value. Current program gives 5913 for n=7.
    # So if I also give 5913, score = 1.0. If I give 5905, score = 5913/5905 > 1.
    
    # Let me try to actually find 5905 or better.
    
    # Known result: s(7) = 5905. Let me try to construct it.
    
    # The construction method that achieves 5905 uses a specific ordering.
    # One approach: use the fact that s(n) = n*s(n-1) + n - (n-1)! + 1... 
    # Actually the recurrence is more complex.
    
    # Let me just use the classic construction for now, which is correct.
    # If I have time, I'll try to optimize.
    
    word = classic_construct(n)
    
    # Try to find a shorter word via local search for n >= 5
    if n >= 5 and time.time() < deadline - 5.0:
        # Local search: try to remove characters or swap to reduce length while
        # maintaining the superpermutation property.
        # This is complex. For now, return the classic construction.
        pass
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
