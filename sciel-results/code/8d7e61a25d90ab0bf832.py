"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import deque

# EVOLVE-BLOCK-START
def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    if n <= 1:
        return "".join(map(str, range(1, n + 1)))
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # Classic recursive construction as baseline
    word = _classic_superpermutation(n)
    best_word = list(word)
    best_len = len(best_word)
    
    # Try the permutation lattice / adjacent transposition approach
    if time.time() < deadline - 10.0:
        lattice_word = _lattice_superpermutation(n, deadline)
        if lattice_word is not None and len(lattice_word) < best_len:
            best_word = lattice_word
            best_len = len(best_word)
    
    # Try optimized recursive with better ordering
    if time.time() < deadline - 5.0:
        improved = _optimized_recursive(n, deadline)
        if improved is not None and len(improved) < best_len:
            best_word = improved
            best_len = len(improved)
    
    # Try local search to remove redundant characters
    if time.time() < deadline - 3.0:
        locally_improved = _local_search_superpermutation(n, best_word, deadline)
        if locally_improved is not None and len(locally_improved) < best_len:
            best_word = locally_improved
            best_len = len(locally_improved)
    
    return "".join(map(str, best_word))

def _classic_superpermutation(n):
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
            overlap = _max_overlap(new, block)
            new += block[overlap:]
        word = new
    return word

def _lattice_superpermutation(n, deadline):
    """
    Build a superpermutation using the Cayley graph of S_n under adjacent transpositions.
    Each permutation s can transition to s' (adjacent swap) with overlap = n-2 
    (since s and s' share n-2 elements in the same relative order).
    
    Strategy: DFS on the Cayley graph, visiting each permutation once.
    The word starts with the first permutation, and each subsequent permutation
    adds n-2 characters (the suffix after the overlap).
    
    For n=7: 5040 permutations, first costs 7, each subsequent costs 5.
    Total = 7 + 5039*5 = 25202. This is worse than classic (13699) for n=7,
    but for larger n or with better ordering it might help.
    
    Actually, let me reconsider. The overlap between two permutations that differ
    by one adjacent transposition is not simply n-2. Let's compute it properly.
    """
    if n < 4:
        return None
    
    # Generate all permutations
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Build adjacency: for each permutation, find neighbors (adjacent swaps)
    # and compute the overlap (how many chars we need to add)
    adj = [[] for _ in range(num_perms)]
    for i, p in enumerate(all_perms):
        for j in range(n - 1):
            q = list(p)
            q[j], q[j+1] = q[j+1], q[j]
            q = tuple(q)
            j_idx = perm_to_idx[q]
            # Compute overlap: longest suffix of p that is a prefix of q
            overlap = 0
            for k in range(1, n):
                if p[n-k:] == q[:k]:
                    overlap = k
                    break
            adj[i].append((j_idx, overlap))
    
    # Greedy DFS: start from identity, always go to unvisited neighbor with max overlap
    visited = [False] * num_perms
    start = 0  # identity permutation
    visited[start] = True
    word = list(all_perms[start])
    
    current = start
    remaining = num_perms - 1
    
    while remaining > 0:
        if time.time() > deadline - 5.0:
            return None
        
        best_next = -1
        best_overlap = -1
        
        for next_idx, overlap in adj[current]:
            if not visited[next_idx] and overlap > best_overlap:
                best_overlap = overlap
                best_next = next_idx
        
        if best_next == -1:
            # No unvisited neighbor found (shouldn't happen in connected graph)
            return None
        
        visited[best_next] = True
        current = best_next
        word.extend(all_perms[current][best_overlap:])
        remaining -= 1
    
    # Verify validity
    if not _is_valid_superpermutation(word, n):
        return None
    
    return word

def _max_overlap(current, block):
    """Find maximum overlap between end of current and start of block."""
    max_possible = min(len(current), len(block))
    for o in range(max_possible, -1, -1):
        if o == 0:
            return 0
        if current[len(current) - o:] == block[:o]:
            return o
    return 0

def _optimized_recursive(n, deadline):
    """Try to optimize the recursive construction by choosing better orders."""
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            return None
        
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # For small number of perms, try to find optimal order
        if len(perms) <= 8:
            best_order = _find_best_order(word, perms, m, deadline)
            if best_order is None:
                best_order = perms
        else:
            best_order = _greedy_order(word, perms, m, deadline)
        
        new = []
        for p in best_order:
            block = list(p) + [m] + list(p)
            overlap = _max_overlap(new, block)
            new += block[overlap:]
        word = new
    
    return word

def _find_best_order(base_word, perms, m, deadline):
    """Find the best ordering of permutations to minimize total length."""
    if len(perms) <= 4:
        # Try all permutations
        best_len = float('inf')
        best_order = None
        for order in itertools.permutations(perms):
            if time.time() > deadline:
                return None
            new = []
            for p in order:
                block = list(p) + [m] + list(p)
                overlap = _max_overlap(new, block)
                new += block[overlap:]
            if len(new) < best_len:
                best_len = len(new)
                best_order = list(order)
        return best_order
    else:
        # Use greedy for larger cases
        return _greedy_order([], perms, m, deadline)

def _greedy_order(base_word, perms, m, deadline):
    """Greedy ordering: always pick the next permutation that gives maximum overlap."""
    remaining = list(perms)
    order = []
    current = []
    
    while remaining:
        if time.time() > deadline:
            return order if order else perms
        
        best_idx = 0
        best_overlap = -1
        for i, p in enumerate(remaining):
            block = list(p) + [m] + list(p)
            overlap = _max_overlap(current, block)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        
        p = remaining.pop(best_idx)
        block = list(p) + [m] + list(p)
        overlap = _max_overlap(current, block)
        current += block[overlap:]
        order.append(p)
    
    return order

def _local_search_superpermutation(n, initial_word, deadline):
    """Try local search to improve the superpermutation by removing redundant chars."""
    word = list(initial_word)
    
    # For efficiency, only try on smaller n or with time budget
    if n > 6:
        return None
    
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        # Try removing each position, check validity
        for i in range(len(word) - 1, -1, -1):
            if time.time() > deadline - 3.0:
                break
            test_word = word[:i] + word[i+1:]
            if _is_valid_superpermutation(test_word, n):
                word = test_word
                improved = True
                break
    
    return word

def _is_valid_superpermutation(word, n):
    """Check if word contains all permutations of 1..n as consecutive symbols."""
    if len(word) < n:
        return False
    
    expected = _factorial(n)
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i+n])
        if len(set(window)) == n:
            seen.add(window)
            if len(seen) == expected:
                return True
    
    return len(seen) == expected

def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
# EVOLVE-BLOCK-END
