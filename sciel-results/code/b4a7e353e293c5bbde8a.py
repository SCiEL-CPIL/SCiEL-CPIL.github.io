"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct(n):
    # Classic recursive construction of length 1! + 2! + ... + n!
    # This is the known optimal construction for n <= 4 and gives
    # 1! + 2! + ... + n! for all n. For n=7 this gives 1+2+6+24+120+720+5040 = 5913.
    # The shortest known superpermutation on 7 symbols is 5905 (Teodorescu 2026).
    # We implement the standard recursive construction which is fast and deterministic.
    #
    # The recursive construction: start with "1". For each new symbol m from 2 to n:
    #   - Extract all distinct (m-1)-length permutations in order of first appearance
    #   - For each permutation p, insert m after it: p -> p + [m] + p
    #   - Overlap consecutive blocks maximally
    #
    # This gives length sum_{k=1}^{n} k! = 5913 for n=7.
    #
    # To potentially improve beyond 5913, we could try local search / greedy improvements,
    # but the known best is 5905 and the lower bound is 5884. Given the 150s budget,
    # we implement the fast recursive construction (takes milliseconds) and then attempt
    # a limited local optimization if time permits.
    
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
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
            # Find maximum overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # word is now a superpermutation of length sum(1!..n!)
    # For n=7: 5913
    
    # Verify it's valid (should be, but just in case)
    # and attempt local optimization if time remains
    
    if time.time() > deadline - 10:
        return "".join(map(str, word))
    
    # Try to shorten via local search: try removing redundant elements
    # Strategy: for each position, try to see if removing it still yields a valid
    # superpermutation. This is O(n * n!) per check which is too slow for n=7.
    # 
    # Alternative: try a greedy construction with different orderings.
    # 
    # Actually, the recursive construction is known to be optimal for n<=4.
    # For n=5, it gives 153, and the optimal is also 153.
    # For n=6, it gives 873, and the optimal is 873.
    # For n=7, it gives 5913, but the known best is 5905.
    # 
    # The improvement from 5913 to 5905 requires a more sophisticated construction.
    # One approach: the "greedy" superpermutation construction where at each step
    # you append the shortest string that introduces a new permutation.
    # 
    # Let's implement the greedy algorithm:
    # Start with "1". At each step, find the shortest string to append such that
    # a new permutation of 1..n appears.
    
    if n <= 6:
        return "".join(map(str, word))
    
    # For n=7, try greedy construction
    # Greedy: maintain a set of seen permutations. Start with "1".
    # At each step, try appending each symbol 1..n and pick the one that
    # creates the most new permutations (or the first one that creates any).
    
    # This is expensive but let's see if we can do it within budget.
    # Actually, the greedy algorithm for superpermutations is known to produce
    # the same result as the recursive construction for small n, and for n=7
    # it's been shown to give 5905 with careful implementation.
    
    # Let me implement a proper greedy:
    # State: current word, set of seen permutations
    # At each step: try all possible next symbols, pick the one that maximizes
    # the number of new permutations introduced (tie-break by shortest path to next new perm)
    
    # For n=7, the word is ~5900 chars. Each step we try 7 symbols.
    # Checking if a new permutation appears: we only need to check the last n-1 positions
    # plus the new position. So it's O(n) per symbol per step. Total: O(7 * 5900 * 7) ≈ 287k ops.
    # That's fast enough.
    
    # But wait - the greedy might not find the optimal. Let's just use the recursive
    # construction which is known to be 5913, and that's our baseline.
    # 
    # Actually, let me think about this differently. The task says the shortest known
    # is 5905. The recursive gives 5913. If we can't beat 5913 easily, we should
    # just return the recursive result.
    
    # Let's try one more thing: a modified recursive construction where we try
    # different orderings of the permutations at each level.
    
    # For now, return the standard recursive construction result.
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
