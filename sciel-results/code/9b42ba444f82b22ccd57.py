"""
Matrix multiplication 3x5x7 tensor decomposition via minimum-cardinality exact cover.

Approach:
1. Enumerate all valid rank-1 tensor supports (combinatorial rectangles) whose
   support is contained within the support of the multiplication tensor.
2. For each valid support, determine the coefficient (must be 1 for exact match
   with binary entries, or potentially half-integers if we allow cancellation).
   Since the target tensor has entries 0 or 1, and we want exact decomposition,
   we restrict to binary coefficients (0 or 1) for simplicity, as half-integers
   would require cancellation which is harder to manage in a simple exact cover.
3. Formulate as a set cover problem: each valid rank-1 term covers a subset of
   the non-zero entries of the tensor. We need to cover each non-zero entry exactly once.
4. Use a greedy heuristic or a simple ILP solver (via scipy if available, otherwise
   a custom backtracking with pruning) to find a minimal set of terms.

Given the constraints (120s limit, small dimensions), we will:
- Enumerate all valid binary rank-1 terms (u, v, w) with small supports.
- Use a greedy set cover to get a good initial solution.
- Try to improve with local search or exact methods if time permits.

However, for 3x5x7, the number of valid terms is manageable. Let's enumerate all
valid (Su, Sv, Sw) such that Su x Sv x Sw is a subset of supp(T).

Key insight: For the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1,
a rank-1 term with support Su x Sv x Sw is valid iff for all a in Su, b in Sv, c in Sw,
T[a,b,c] = 1.

We can precompute for each pair (a, b) the set of c such that T[a,b,c]=1, and similarly
for other pairs.

Let's implement a greedy set cover:
1. List all valid rank-1 terms (as sets of covered indices in the flattened tensor).
2. Greedily pick the term that covers the most uncovered elements.
3. Repeat until all elements are covered.

This will give a valid decomposition. The rank is the number of terms.

To potentially improve, we can try to remove redundant terms or swap terms.
But for now, greedy should give a rank close to the schoolbook (105) or better.

Actually, let's think about the structure. The tensor has 105 non-zero entries.
Each rank-1 term covers a product of the sizes of its supports.
If we use terms with support size 1 (i.e., u, v, w each have one non-zero entry),
we get 105 terms.

Can we do better? Yes, if we find terms that cover multiple entries.
For example, if we fix i and k, and vary j, we get a term that covers m entries?
Let's check: Fix i, k. Let Su = {i*m + j for j in range(m)}, Sv = {j*p + k for j in range(m)}, Sw = {k*n + i}.
Then for each j, T[i*m+j, j*p+k, k*n+i] = 1. So this is a valid rank-1 term with
|Su|=m, |Sv|=m, |Sw|=1. It covers m entries.

Similarly, fix i, j, vary k: covers p entries.
Fix j, k, vary i: covers n entries.

So we can group by (i,k) to cover m entries at a time, etc.

Let's enumerate all such structured terms and use greedy set cover.

Steps:
1. Generate all valid rank-1 terms with binary coefficients.
   - Terms of type (i,k): fix i, k, vary j. Covers {(i*m+j, j*p+k, k*n+i) for j in range(m)}.
   - Terms of type (i,j): fix i, j, vary k. Covers {(i*m+j, j*p+k, k*n+i) for k in range(p)}.
   - Terms of type (j,k): fix j, k, vary i. Covers {(i*m+j, j*p+k, k*n+i) for i in range(n)}.
   - Also consider smaller terms, but these are the largest structured ones.
   - We can also consider combinations, but let's start with these.

2. Use greedy set cover to select terms.

3. Return the U, V, W matrices.

Let's implement this.
"""

import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    N = n * m
    M = m * p
    P = p * n
    
    # Total number of non-zero entries in the tensor
    total_entries = n * m * p
    
    # Generate all valid rank-1 terms as sets of covered (a, b, c) indices
    terms = []
    
    # Type 1: Fix i, k, vary j
    for i in range(n):
        for k in range(p):
            covered = set()
            for j in range(m):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                covered.add((a, b, c))
            terms.append(covered)
    
    # Type 2: Fix i, j, vary k
    for i in range(n):
        for j in range(m):
            covered = set()
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                covered.add((a, b, c))
            terms.append(covered)
    
    # Type 3: Fix j, k, vary i
    for j in range(m):
        for k in range(p):
            covered = set()
            for i in range(n):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                covered.add((a, b, c))
            terms.append(covered)
    
    # Also add individual entries (rank-1 terms with support size 1) as fallback
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                terms.append({(a, b, c)})
    
    # Greedy set cover
    all_elements = set()
    for term in terms:
        all_elements.update(term)
    
    uncovered = set(all_elements)
    selected_terms = []
    
    # Sort terms by size descending for greedy
    terms_sorted = sorted(terms, key=len, reverse=True)
    
    while uncovered:
        best_term = None
        best_covered = 0
        for term in terms_sorted:
            covered = len(term & uncovered)
            if covered > best_covered:
                best_covered = covered
                best_term = term
            if best_covered == len(uncovered):
                break
        if best_term is None or best_covered == 0:
            # Should not happen if terms cover all elements
            break
        selected_terms.append(best_term)
        uncovered -= best_term
    
    # Build U, V, W matrices
    rank = len(selected_terms)
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    
    for col, term in enumerate(selected_terms):
        # For each (a, b, c) in term, set U[a][col]=1, V[b][col]=1, W[c][col]=1
        for (a, b, c) in term:
            U[a][col] = 1
            V[b][col] = 1
            W[c][col] = 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
