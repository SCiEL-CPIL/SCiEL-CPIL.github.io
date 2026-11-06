import sys
import functools

N = 18

# We need to generate the network based on the optimal splits found by DP.
# We'll store the best split for each n.

best_split = {}

@functools.lru_cache(maxsize=None)
def merge_size(n, m):
    """
    Calculate the number of comparators in a Batcher odd-merge of n and m.
    """
    if n == 0 or m == 0:
        return 0
    if n == 1 and m == 1:
        return 1
    
    n_odd = n // 2
    n_even = n - n_odd
    m_odd = m // 2
    m_even = m - m_odd
    
    # The odd-merge algorithm splits into:
    # 1. Merge(n_odd, m_even)
    # 2. Merge(n_even, m_odd)
    # 3. (n+m)//2 final comparators
    return merge_size(n_odd, m_even) + merge_size(n_even, m_odd) + (n + m) // 2

@functools.lru_cache(maxsize=None)
def sort_size(n):
    """
    Calculate the minimum number of comparators for a Batcher merge sort of n wires.
    """
    if n <= 1:
        return 0
    
    min_cost = float('inf')
    best_k = 1
    
    # Try all splits k and n-k
    for k in range(1, n):
        # Symmetry: sorting k and n-k is same cost as n-k and k?
        # merge_size(k, n-k) == merge_size(n-k, k) is NOT necessarily true in terms of internal structure
        # but the total size C(k) + C(n-k) + M(k, n-k) is what we care about.
        # Note: M(k, n-k) is the cost to merge a sorted list of k and sorted list of n-k.
        # The merge algorithm itself is not symmetric in implementation (odd/even split depends on sizes),
        # but the *count* of comparators M(k, m) should be symmetric?
        # Let's check: M(1, 2).
        # n=1, m=2. n_odd=0, n_even=1. m_odd=1, m_even=1.
        # M(0,1) + M(1,1) + 1 = 0 + 1 + 1 = 2.
        # M(2, 1). n=2, m=1. n_odd=1, n_even=1. m_odd=0, m_even=1.
        # M(1,1) + M(1,0) + 1 = 1 + 0 + 1 = 2.
        # Yes, symmetric.
        
        cost = sort_size(k) + sort_size(n - k) + merge_size(k, n - k)
        if cost < min_cost:
            min_cost = cost
            best_k = k
            
    best_split[n] = best_k
    return min_cost

def generate_network(n, offset=0, comps=None):
    """
    Recursively generate the comparators for the optimal Batcher network.
    """
    if comps is None:
        comps = []
        
    if n <= 1:
        return comps
        
    k = best_split[n]
    
    # Sort left half
    generate_network(k, offset, comps)
    
    # Sort right half
    generate_network(n - k, offset + k, comps)
    
    # Merge the two halves
    generate_merge(k, n - k, offset, comps)
    
    return comps

def generate_merge(n, m, offset, comps):
    """
    Generate comparators for Batcher odd-merge of n (at offset) and m (at offset+n).
    """
    if n == 0 or m == 0:
        return
    
    if n == 1 and m == 1:
        comps.append((offset, offset + 1))
        return
        
    n_odd = n // 2
    n_even = n - n_odd
    m_odd = m // 2
    m_even = m - m_odd
    
    # 1. Merge A_odd with B_even
    # A_odd are at indices: 1, 3, ... in the first half?
    # In Batcher odd merge:
    # A = A_odd A_even (interleaved? No, A is sorted. We take odd positions of A?)
    # Standard Batcher Odd Merge:
    # Split A into A_odd (1st, 3rd, ...) and A_even (2nd, 4th, ...).
    # Split B into B_odd (1st, 3rd, ...) and B_even (2nd, 4th, ...).
    # Merge A_odd with B_even.
    # Merge A_even with B_odd.
    # Then compare adjacent pairs in the combined list.
    
    # The mapping of indices:
    # A_odd elements are at indices 1, 3, ..., 2*n_odd-1 in A (0-indexed).
    # Global indices: offset + 1, offset + 3, ...
    # B_even elements are at indices 2, 4, ..., 2*m_even in B (0-indexed in B).
    # Global indices: (offset+n) + 1, (offset+n) + 3, ... ?
    # Wait, B_odd is 1st, 3rd... of B. B_even is 2nd, 4th... of B.
    
    # Let's map the local indices of the sub-merges to global indices.
    
    # Sub-merge 1: A_odd (size n_odd) and B_even (size m_even).
    # A_odd local index i -> global index offset + 2*i + 1.
    # B_even local index j -> global index offset + n + 2*j + 1.
    # We need to generate a merge network for sizes n_odd and m_even.
    # The "first" list is A_odd, the "second" is B_even.
    # They are not contiguous in memory, so we can't just call generate_merge with a simple offset.
    # We need to map the comparators generated for a contiguous merge of size n_odd+m_even
    # to the specific wires.
    
    # Helper to map a contiguous merge comparator (i, j) to global wires
    # where the first list (size n1) maps to indices f1(k) and second (size n2) maps to f2(k).
    
    def map_merge_comps(n1, n2, f1, f2):
        sub_comps = generate_merge_flat(n1, n2)
        mapped = []
        for i, j in sub_comps:
            if i < n1:
                gi = f1(i)
            else:
                gi = f2(i - n1)
                
            if j < n1:
                gj = f1(j)
            else:
                gj = f2(j - n1)
                
            if gi > gj:
                gi, gj = gj, gi
            mapped.append((gi, gj))
        return mapped

    # Define mappings for Sub-merge 1: A_odd (n_odd) and B_even (m_even)
    def f_a_odd(i):
        return offset + 2 * i + 1
        
    def f_b_even(j):
        return offset + n + 2 * j + 1
        
    comps.extend(map_merge_comps(n_odd, m_even, f_a_odd, f_b_even))

    # Sub-merge 2: A_even (n_even) and B_odd (m_odd)
    # A_even local index i -> global index offset + 2*i.
    # B_odd local index j -> global index offset + n + 2*j.
    
    def f_a_even(i):
        return offset + 2 * i
        
    def f_b_odd(j):
        return offset + n + 2 * j
        
    comps.extend(map_merge_comps(n_even, m_odd, f_a_even, f_b_odd))
    
    # Final comparators: Compare adjacent elements in the combined list of size n+m.
    # The combined list is formed by placing the results of the sub-merges back?
    # No, Batcher's odd merge works in place on the interleaved wires?
    # Actually, the standard description is:
    # 1. Merge A_odd and B_even.
    # 2. Merge A_even and B_odd.
    # 3. Compare (0,1), (2,3), ... in the full list.
    
    # The wires involved in step 3 are all wires from offset to offset+n+m-1.
    # So we just add (offset + 2*i, offset + 2*i + 1) for i in range((n+m)//2).
    
    total = n + m
    for i in range(total // 2):
        comps.append((offset + 2 * i, offset + 2 * i + 1))


def generate_merge_flat(n, m):
    """
    Generate comparators for a merge of n and m, assuming they are contiguous starting at 0.
    Returns list of (i, j) where 0 <= i < j < n+m.
    """
    if n == 0 or m == 0:
        return []
    if n == 1 and m == 1:
        return [(0, 1)]
        
    n_odd = n // 2
    n_even = n - n_odd
    m_odd = m // 2
    m_even = m - m_odd
    
    comps = []
    
    # Merge A_odd (size n_odd) with B_even (size m_even)
    # In the contiguous model:
    # A_odd occupies indices 1, 3, ... of the first half? No.
    # The "odd merge" algorithm assumes A and B are just sequences.
    # When mapping to a contiguous array of size n+m:
    # A is indices 0..n-1. B is indices n..n+m-1.
    # A_odd elements are at 1, 3, ... n-1 (if n odd) or n-2?
    # A_odd = A[1], A[3]...
    # B_even = B[1], B[3]... (indices 1,3... in B, so n+1, n+3 in global)
    
    # This flat generation is tricky because the sub-merges operate on non-contiguous subsets.
    # It is easier to use the mapping function as defined in generate_merge.
    # So generate_merge_flat should probably just call a helper that returns local indices
    # for a "virtual" contiguous merge, which is what generate_merge does via mapping.
    # But generate_merge needs the mappings.
    # For the base case of generate_merge_flat, we just return the comparators for a contiguous block.
    # The recursion in generate_merge handles the mapping.
    # So generate_merge_flat is only needed for the TOP level if we wanted contiguous output?
    # No, generate_merge is the core.
    
    # We can implement generate_merge_flat by calling a helper that returns local comparators
    # for a merge where A is 0..n-1 and B is n..n+m-1, but the recursive step
    # must map correctly.
    
    # Actually, the `map_merge_comps` inside `generate_merge` calls `generate_merge_flat`.
    # `generate_merge_flat` should generate comparators for a standard Batcher merge
    # on a contiguous array of size n+m, where A is first n and B is next m.
    
    # Let's implement generate_merge_flat correctly using the same logic.
    
    # Sub-merge 1: A_odd and B_even.
    # In the contiguous array:
    # A_odd elements are at indices 1, 3, ..., 2*n_odd-1.
    # B_even elements are at indices n+1, n+3, ..., n+2*m_even-1.
    # We need to generate a merge of size n_odd + m_even, and map it to these indices.
    
    # This creates a circular dependency if generate_merge_flat calls generate_merge.
    # Let's define a unified recursive generator that takes mappings.
    
    # Let's refactor:
    # define a function `gen(n, m, mapA, mapB)` that returns global comparators.
    
    return [] # Placeholder, will be replaced by refactored logic below

def construct():
    # Pre-compute sizes and best splits
    for i in range(1, N + 1):
        sort_size(i)
        
    # Generate the network
    network = generate_network(N)
    
    # Verify no duplicates or invalid comparators?
    # The task says "Maximum 4096 comparators".
    # The network size should be around 90.
    
    return network

# Refactored Generator Logic

def gen_merge(n, m, mapA, mapB):
    """
    Generate comparators for merging n elements (mapped by mapA) and m elements (mapped by mapB).
    mapA: list or function of size n, giving global indices for A[0..n-1]
    mapB: list or function of size m, giving global indices for B[0..m-1]
    Returns list of (i, j) global comparators.
    """
    if n == 0 or m == 0:
        return []
        
    if n == 1 and m == 1:
        i = mapA[0]
        j = mapB[0]
        if i > j: i, j = j, i
        return [(i, j)]
        
    n_odd = n // 2
    n_even = n - n_odd
    m_odd = m // 2
    m_even = m - m_odd
    
    comps = []
    
    # 1. Merge A_odd and B_even
    # A_odd: A[1], A[3]...
    mapA_odd = [mapA[2*i + 1] for i in range(n_odd)]
    # B_even: B[1], B[3]...
    mapB_even = [mapB[2*j + 1] for j in range(m_even)]
    
    comps.extend(gen_merge(n_odd, m_even, mapA_odd, mapB_even))
    
    # 2. Merge A_even and B_odd
    # A_even: A[0], A[2]...
    mapA_even = [mapA[2*i] for i in range(n_even)]
    # B_odd: B[0], B[2]...
    mapB_odd = [mapB[2*j] for j in range(m_odd)]
    
    comps.extend(gen_merge(n_even, m_odd, mapA_even, mapB_odd))
    
    # 3. Final comparators
    # The combined list is formed by the "slots" in the original interleaving?
    # No, Batcher's odd merge final step compares adjacent elements in the full list.
    # The full list is the union of wires mapped by mapA and mapB?
    # Actually, the final step operates on the wires at positions 0,1; 2,3; ... in the
    # "virtual" contiguous layout of the merge?
    # Wait, the standard Batcher odd merge produces a sorted list of n+m elements.
    # The elements are in the wires provided by mapA and mapB.
    # The final comparators are between adjacent positions in the sorted output?
    # No, the final comparators are between specific pairs of wires in the input layout.
    # In the standard contiguous layout (A=0..n-1, B=n..n+m-1), the final comparators
    # are (0,1), (2,3), ..., (n+m-2, n+m-1).
    # We need to map these local indices to the global wires.
    
    # The local index k in the final list corresponds to:
    # if k < n, it's A[k] -> mapA[k]
    # if k >= n, it's B[k-n] -> mapB[k-n]
    
    total = n + m
    for i in range(total // 2):
        idx1 = 2 * i
        idx2 = 2 * i + 1
        
        if idx1 < n:
            g1 = mapA[idx1]
        else:
            g1 = mapB[idx1 - n]
            
        if idx2 < n:
            g2 = mapA[idx2]
        else:
            g2 = mapB[idx2 - n]
            
        if g1 > g2:
            g1, g2 = g2, g1
        comps.append((g1, g2))
        
    return comps

# Re-define generate_network to use gen_merge

def generate_network(n, offset=0, comps=None):
    if comps is None:
        comps = []
    if n <= 1:
        return comps
        
    k = best_split[n]
    
    generate_network(k, offset, comps)
    generate_network(n - k, offset + k, comps)
    
    # Merge
    mapA = [offset + i for i in range(k)]
    mapB = [offset + k + i for i in range(n - k)]
    
    comps.extend(gen_merge(k, n - k, mapA, mapB))
    
    return comps

def construct():
    for i in range(1, N + 1):
        sort_size(i)
    network = generate_network(N)
    return network
