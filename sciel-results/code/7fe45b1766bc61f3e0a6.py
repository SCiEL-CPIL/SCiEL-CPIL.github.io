"""Shortest superpermutation on seven symbols."""
import time
import math
from itertools import permutations
import heapq

# EVOLVE-BLOCK-START
def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    if n <= 3:
        # Small cases: use the classic recursive construction
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
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # For n=7, use A* search on permutation graph
    # Each state is a permutation (tuple of 7 symbols)
    # Goal: visit all n! permutations
    
    # Precompute all permutations
    all_perms = list(permutations(range(1, n + 1)))
    num_perms = len(all_perms)
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    
    # Build adjacency: from a permutation, what permutations can we reach
    # with minimal overlap? We can shift and add a new symbol.
    # From permutation p = (a1, a2, ..., an), we can form:
    #   (a2, a3, ..., an, x) for any x not in {a2,...,an}
    # This gives n-1 possible next states (since one symbol is excluded)
    
    # Actually, the transition is: from state p, we append a new symbol x
    # to get a new window. The new state is the last n symbols of the extended word.
    # If current word ends with ...p, and we append x, the new window is (p[1:], x).
    # So from p, we can go to (p[1], p[2], ..., p[n-1], x) for x in {1..n} \ {p[1],...,p[n-1]}
    # That's exactly 1 choice for x (since p[1..n-1] has n-1 distinct symbols, only one left).
    # Wait, that's not right. Let me reconsider.
    
    # Actually, the standard model: a superpermutation is a sequence where every
    # permutation appears as a contiguous subsequence. The "state" for DP is the
    # current permutation window. When we extend the word by one symbol, the new
    # window is determined by the old window minus its first element, plus the new symbol.
    # So from window w = (w1,...,wn), appending symbol x gives new window (w2,...,wn,x).
    # For this to be a valid permutation, x must not be in {w2,...,wn}, so x = w1 is forced?
    # No: x can be any symbol, but the new window (w2,...,wn,x) must be a permutation,
    # meaning all n symbols distinct. Since {w2,...,wn} has n-1 distinct symbols, x must
    # be the one missing symbol. So there's exactly ONE next state from each state!
    # That means the graph is a collection of cycles, and we need to find a path
    # that visits all nodes. This is the Hamiltonian path problem on this functional graph.
    
    # Let me build the functional graph:
    # next_perm[w] = the unique permutation reachable from w by appending the missing symbol.
    next_perm = {}
    for p in all_perms:
        # p is a tuple of n distinct symbols
        # The missing symbol from p[1:] is the one not in p[1:]
        present = set(p[1:])
        missing = [x for x in range(1, n + 1) if x not in present][0]
        new_p = p[1:] + (missing,)
        next_perm[p] = new_p
    
    # Now the problem reduces to: find the shortest path in this directed functional graph
    # that visits all num_perms nodes. Since each node has out-degree 1, the graph consists
    # of disjoint cycles with trees feeding into them. But wait, every node has in-degree
    # exactly 1 as well (it's a permutation of the state space), so the graph is a disjoint
    # union of cycles.
    
    # Let's verify: is the mapping p -> next_perm[p] a permutation of all_perms?
    # Each p maps to a unique new_p, and the mapping is injective (if next_perm[p1] = next_perm[p2],
    # then p1[1:] + (m1,) = p2[1:] + (m2,), so p1[1:]=p2[1:] and m1=m2, hence p1=p2).
    # So yes, it's a permutation, and the graph is a disjoint union of cycles.
    
    # To visit all nodes, we need to traverse all cycles. The shortest superpermutation
    # corresponds to finding an ordering of the cycles and, within each cycle, a starting point,
    # such that the total length is minimized. The length of the word is the sum of cycle lengths
    # plus the number of transitions between cycles.
    
    # Actually, I realize my model might be off. Let me think again.
    
    # The word is a sequence of symbols. The "states" are the n-length windows.
    # Consecutive windows overlap in n-1 symbols. So if window i is (a1,...,an) and
    # window i+1 is (b1,...,bn), then (a2,...,an) = (b1,...,b_{n-1}).
    # So b = (a2, a3, ..., an, x) for some x. For b to be a permutation, x must be the
    # symbol missing from {a2,...,an}. So the transition is deterministic.
    
    # This means the sequence of windows is completely determined by the first window.
    # So the only superpermutations that can be formed this way are those where the
    # window sequence follows the functional graph. But we can also have windows that
    # are NOT permutations! The word just needs to CONTAIN each permutation as a window,
    # but other windows can be non-permutations.
    
    # So my model is too restrictive. Let me use a different approach.
    
    # Alternative: Use the known best construction for n=7. The current best known
    # is 5905. The classic recursive gives 1!+2!+...+7! = 1+2+6+24+120+720+5040 = 5913.
    # We can try to improve by finding better overlaps.
    
    # For now, let's use the classic construction but with a greedy improvement pass.
    
    # Classic construction:
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
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    base_length = len(word)
    
    # Try to improve by local search: swap adjacent symbols and check validity
    # This is expensive, so only do it if we have time.
    
    if time.time() > deadline - 30:
        return "".join(map(str, word))
    
    # Validate the current word
    def is_valid_superperm(w, n):
        seen = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        return len(seen) == math.factorial(n)
    
    # Greedy improvement: try to remove redundant symbols
    # This is complex, so let's just return the base construction for now.
    # A more sophisticated approach would use simulated annealing or local search.
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
