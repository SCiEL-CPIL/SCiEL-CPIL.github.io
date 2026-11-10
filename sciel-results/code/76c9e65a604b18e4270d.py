"""Shortest superpermutation search with deadline guard."""
import time
import sys
from itertools import permutations

# EVOLVE-BLOCK-START

def construct(n):
    if n <= 1:
        return "1"
    
    # Deadline guard: 120s limit, keep 10s safety margin
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 15.0
    
    # For small n, we can compute exact shortest superpermutations
    # Known lengths: n=1:1, n=2:3, n=3:9, n=4:33, n=5:153, n=6:873, n=7:5905
    # The classic formula gives 1!+2!+...+n! which is optimal for n<=5
    # For n=6 and n=7, the greedy method gives suboptimal results
    
    # Strategy: Use a greedy approach with backtracking for better overlap
    # The key insight is that when inserting new symbols, we want to maximize overlap
    
    # For n <= 5, use the classic construction (it's optimal)
    if n <= 5:
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
                max_overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0:
                        max_overlap = 0
                        break
                    if new[len(new) - o:] == block[:o]:
                        max_overlap = o
                        break
                new += block[max_overlap:]
            word = new
        return "".join(map(str, word))
    
    # For n = 6, 7: Use a more sophisticated greedy with multiple passes
    # The classic recursive construction is the baseline.
    # We can try to improve by reordering the permutations at each level.
    
    # Let's implement the standard greedy but with a twist:
    # At each step, when we have a list of (m-1)-permutations to expand,
    # we can try different orderings to minimize total length.
    
    # Actually, the standard approach for shortest superpermutation uses
    # the fact that the problem is equivalent to finding a shortest path
    # in a de Bruijn-like graph where nodes are (n-1)-permutations and
    # edges represent adding a new symbol.
    
    # For n=7, 5040 permutations is manageable for a greedy heuristic.
    
    # Let's use a different approach: build the word incrementally
    # using a greedy algorithm that at each step chooses the next symbol
    # that covers the most uncovered permutations.
    
    # Actually, let me think about this differently.
    # A superpermutation of length L over n symbols contains all n! permutations
    # as consecutive windows. The number of windows is L - n + 1.
    # We need L - n + 1 >= n!, so L >= n! + n - 1.
    # For n=7: 5040 + 6 = 5046 is the theoretical minimum (but not achievable).
    
    # The known best for n=7 is 5905.
    # The classic construction gives 1!+2!+3!+4!+5!+6!+7! = 1+2+6+24+120+720+5040 = 5913.
    # So we need to save 8 characters to get from 5913 to 5905.
    
    # Let me implement the classic construction first and see if we can optimize.
    
    def classic_construction(n):
        word = [1]
        for m in range(2, n + 1):
            # Find all distinct (m-1)-permutations in the current word
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            # For each permutation, insert m in the middle
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find maximum overlap with current new
                max_overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0:
                        max_overlap = 0
                        break
                    if new[len(new) - o:] == block[:o]:
                        max_overlap = o
                        break
                new += block[max_overlap:]
            word = new
        return word
    
    # Try classic construction
    word = classic_construction(n)
    current_len = len(word)
    
    # Try to improve by reordering permutations at each level
    # This is complex, so let's try a simpler improvement:
    # At each level, try different orderings of the permutations
    
    # For now, return the classic construction result
    # which gives 5913 for n=7
    
    # Check if we're within time limit
    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
        return "".join(map(str, word))
    
    # Try a local search to reduce length
    # One approach: try to find a shorter ordering of the permutation blocks
    
    # Actually, let's try a different greedy: at each step, choose the
    # next (m-1)-permutation that maximizes overlap with the current end
    
    def greedy_with_reorder(n):
        word = [1]
        for m in range(2, n + 1):
            # Find all distinct (m-1)-permutations
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            # Build blocks
            blocks = {}
            for p in perms:
                blocks[p] = list(p) + [m] + list(p)
            
            # Greedily add blocks in order of maximum overlap
            new = []
            remaining = set(perms)
            
            while remaining:
                best_p = None
                best_overlap = -1
                for p in remaining:
                    block = blocks[p]
                    # Find maximum overlap with current new
                    max_ov = 0
                    for o in range(min(len(new), len(block)), -1, -1):
                        if o == 0:
                            max_ov = 0
                            break
                        if new[len(new) - o:] == block[:o]:
                            max_ov = o
                            break
                    if max_ov > best_overlap:
                        best_overlap = max_ov
                        best_p = p
                
                if best_p is None:
                    break
                    
                block = blocks[best_p]
                new += block[best_overlap:]
                remaining.remove(best_p)
            
            word = new
        return word
    
    # Try the greedy with reorder
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        word2 = greedy_with_reorder(n)
        if len(word2) < current_len:
            word = word2
            current_len = len(word2)
    
    return "".join(map(str, word))

# EVOLVE-BLOCK-END
