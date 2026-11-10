"""Superpermutation solver using a de Bruijn graph search on (n-1)-permutations."""
import time
import math
import random
from heapq import heappush, heappop
from itertools import permutations

# Global deadline guard
_START_TIME = None
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0  # seconds

def _remaining_time():
    if _START_TIME is None:
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _is_permutation(seq, n):
    """Check if seq is a permutation of 1..n."""
    if len(seq) != n:
        return False
    return len(set(seq)) == n

def construct(n):
    global _START_TIME
    _START_TIME = time.time()
    
    # For small n, use exact known constructions or brute force
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Known shortest superpermutation for n=4 is length 33
        # 1! + 2! + 3! + 4! = 1+2+6+24 = 33
        return _classic_construction(n)
    
    # For n >= 5, the classic construction gives length 1!+2!+...+n!
    # which is not optimal but is a valid upper bound.
    # We'll try to improve using a graph-based approach.
    
    if n == 5:
        # Known shortest for n=5 is 152 (vs classic 153)
        # Let's try to find it or fall back to classic
        result = _try_graph_search(n)
        if result is not None:
            return result
        return _classic_construction(n)
    
    if n >= 6:
        # For larger n, graph search is infeasible. Use classic construction
        # but try a few improvements with local search if time permits
        result = _classic_construction(n)
        # Try to improve with local search if time allows
        if _remaining_time() > 5.0:
            improved = _local_search_improve(result, n)
            if improved is not None:
                return improved
        return result
    
    return _classic_construction(n)

def _classic_construction(n):
    """The classic recursive construction of length 1! + 2! + ... + n!."""
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
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def _try_graph_search(n):
    """
    Try to find a shorter superpermutation using a graph-based approach.
    Nodes are (n-1)-permutations, edges represent appending a symbol.
    We use a greedy best-first search with heuristic.
    """
    if n > 5:
        return None  # Too many nodes for n>5
    
    # For n=5, there are 4! = 24 nodes (permutations of 4 elements)
    # We want to find a path that visits all 5! = 120 permutations of 5 elements
    
    # Actually, let's think about this differently.
    # A superpermutation of length L contains n! windows of length n.
    # Each window is a permutation of 1..n.
    # Consecutive windows overlap in n-1 positions.
    # So if window i is (a1, a2, ..., an) and window i+1 is (b1, b2, ..., bn),
    # then (a2, ..., an) = (b1, ..., b_{n-1}).
    
    # This means we can model this as a path in a graph where:
    # - Nodes are (n-1)-tuples that are permutations of n-1 distinct elements from 1..n
    # - Actually, nodes are (n-1)-permutations of some (n-1)-subset of 1..n
    # - An edge from node u to node v exists if v[0] = u[1], ..., v[n-2] = u[n-2]
    #   Wait, that's not right either.
    
    # Let me reconsider. If we have a word W of length L, the windows are W[0:n], W[1:n+1], ..., W[L-n:n].
    # Window i and window i+1 share n-1 symbols: W[i+1:i+n] and W[i+2:i+n+1].
    # So the transition from window i to window i+1 is determined by the last n-1 symbols of window i
    # and the first symbol of window i+1.
    
    # Actually, the state can be represented by the current (n-1)-suffix of the word.
    # When we append a new symbol x, the new (n-1)-suffix is the last n-2 symbols of the old suffix plus x.
    
    # But we also need to track which n-permutations we've already seen.
    # This is a huge state space, so we can't do exact search for n=5 either in general.
    
    # For n=5, let's just use the known optimal construction or try a few heuristics.
    # The known shortest superpermutation for n=5 has length 152.
    
    # Let's try a randomized greedy approach:
    # Start with a random permutation of 1..5
    # Then repeatedly append a symbol that maximizes overlap with unvisited permutations
    
    best_result = None
    best_len = float('inf')
    
    for trial in range(50):
        if _remaining_time() < 5.0:
            break
        
        # Start with a random permutation
        start = list(range(1, n + 1))
        random.shuffle(start)
        
        word = list(start)
        visited = {tuple(start)}
        
        # Greedily extend
        while len(visited) < math.factorial(n):
            if _remaining_time() < 5.0:
                break
            
            # Find all possible next symbols
            # The last n-1 symbols of the current word determine which permutations we can form next
            last_n_minus_1 = tuple(word[-(n-1):])
            
            # For each symbol x not in last_n_minus_1, we can form a new permutation
            # by appending x and checking if the new window is a valid permutation
            candidates = []
            for x in range(1, n + 1):
                if x not in last_n_minus_1:
                    new_window = last_n_minus_1[1:] + (x,)
                    # This is a permutation of n-1 elements, but we need to check
                    # if it's a valid (n-1)-permutation that can be extended
                    # Actually, any (n-1)-tuple of distinct elements from 1..n is valid
                    if len(set(new_window)) == n - 1:
                        # Check if this leads to a new n-permutation
                        # The new n-permutation would be last_n_minus_1 + (x,)
                        # Wait, no. The new window is word[-(n-1):] + [x], which has length n.
                        # But word[-(n-1):] already has n-1 distinct elements, so adding x
                        # (which is not in last_n_minus_1) gives n distinct elements.
                        full_window = last_n_minus_1 + (x,)
                        if full_window not in visited:
                            candidates.append((x, full_window))
            
            if not candidates:
                break
            
            # Pick the candidate that gives the most future options
            # Heuristic: prefer symbols that leave more options for the next step
            best_candidate = None
            best_score = -1
            for x, fw in candidates:
                # Score based on how many new permutations we can reach from this state
                score = 0
                # Count how many new permutations we could form from the new state
                new_last_n_minus_1 = fw[1:]
                for y in range(1, n + 1):
                    if y not in new_last_n_minus_1:
                        next_window = new_last_n_minus_1[1:] + (y,)
                        if len(set(next_window)) == n - 1:
                            full_next = new_last_n_minus_1 + (y,)
                            if full_next not in visited:
                                score += 1
                if score > best_score:
                    best_score = score
                    best_candidate = (x, fw)
            
            if best_candidate is None:
                break
            
            x, fw = best_candidate
            word.append(x)
            visited.add(fw)
        
        if len(visited) == math.factorial(n) and len(word) < best_len:
            best_len = len(word)
            best_result = "".join(map(str, word))
    
    return best_result

def _local_search_improve(word_str, n):
    """Try to improve the word using local search."""
    # This is a placeholder for now
    return None

# Test
if __name__ == "__main__":
    for n in range(1, 6):
        result = construct(n)
        print(f"n={n}: length={len(result)}, word={result[:50]}...")
