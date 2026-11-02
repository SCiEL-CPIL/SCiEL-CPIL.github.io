import time
import random
import numpy as np
from fractions import Fraction
from itertools import product, combinations

N, M, P = 2, 3, 10
DEADLINE = 140.0
START = time.time()

def _elapsed():
    return time.time() - START

def _check_deadline():
    return _elapsed() > DEADLINE

def _tensor_entries():
    return [(i, j, k) for i in range(N) for j in range(M) for k in range(P)]

def _build_strassen_rank55():
    rank = 55
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    a00, a01, a10, a11 = 0, 1, 3, 4
    a02, a12 = 2, 5
    for b in range(5):
        k0, k1 = 2 * b, 2 * b + 1
        b00, b01 = k0, k1
        b10, b11 = 10 + k0, 10 + k1
        c00, c01 = 2 * k0, 2 * k1
        c10, c11 = 2 * k0 + 1, 2 * k1 + 1
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1
        col += 1
    for k in range(P):
        b2k = 2 * P + k
        U[a02, col] += 1; V[b2k, col] += 1; W[2 * k, col] += 1
        col += 1
        U[a12, col] += 1; V[b2k, col] += 1; W[2 * k + 1, col] += 1
        col += 1
    return U, V, W

def _verify_np(U, V, W, tol=1e-9):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = 0.0
                for r in range(rank):
                    s += U[i * M + j, r] * V[j * P + k, r] * W[k * N + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_exact(U, V, W):
    Uf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in U]
    Vf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in V]
    Wf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in W]
    rank = len(Uf[0])
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = Fraction(0)
                for r in range(rank):
                    s += Uf[i * M + j][r] * Vf[j * P + k][r] * Wf[k * N + i][r]
                if s != 1:
                    return False
    return True

def _strassen_support():
    support = []
    for b in range(5):
        k0, k1 = 2 * b, 2 * b + 1
        support.append({(0, 0, k0), (1, 1, k1)})
        support.append({(1, 0, k0), (1, 1, k1)})
        support.append({(0, 1, k1), (1, 1, k1)})
        support.append({(1, 1, k0), (1, 0, k0)})
        support.append({(0, 0, k0), (0, 1, k1)})
        support.append({(1, 1, k1)})
        support.append({(0, 0, k0)})
    for k in range(P):
        support.append({(0, 2, k)})
        support.append({(1, 2, k)})
    return support

def _precompute_coverage(support):
    """Precompute for each entry, the set of terms that cover it."""
    entries = _tensor_entries()
    entry_idx = {e: idx for idx, e in enumerate(entries)}
    num_entries = len(entries)
    num_terms = len(support)
    # coverage[entry_idx] = list of term indices covering this entry
    coverage = [[] for _ in range(num_entries)]
    for r, s in enumerate(support):
        for e in s:
            if e in entry_idx:
                coverage[entry_idx[e]].append(r)
    return coverage, entry_idx, num_entries

def _solve_support(support, rank):
    entries = _tensor_entries()
    entry_idx = {e: idx for idx, e in enumerate(entries)}
    num_entries = len(entries)
    num_vars = 3 * rank
    A = np.zeros((num_entries, num_vars))
    b = np.ones(num_entries)
    for r, term_entries in enumerate(support):
        for (i, j, k) in term_entries:
            if (i, j, k) not in entry_idx:
                continue
            eq = entry_idx[(i, j, k)]
            A[eq, r] = 1.0
            A[eq, rank + r] = 1.0
            A[eq, 2 * rank + r] = 1.0
    try:
        sol, residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if np.max(np.abs(A @ sol - b)) > 1e-8:
        return None
    U = sol[:rank].reshape(rank, N * M).T
    V = sol[rank:2 * rank].reshape(rank, M * P).T
    W = sol[2 * rank:].reshape(rank, P * N).T
    return U, V, W

def _search_rank54():
    if _check_deadline():
        return None
    base_support = _strassen_support()
    coverage, entry_idx, num_entries = _precompute_coverage(base_support)
    
    # Precompute: for each pair (r1, r2), compute the set of entries
    # that are covered ONLY by r1 or r2 (not by any other term)
    # This is the "unique_to_pair" set
    
    # For efficiency, precompute for each entry the count of covering terms
    cover_count = [len(c) for c in coverage]
    
    # For each entry, if it's covered by exactly 2 terms, record which pair
    # An entry is "unique to pair (r1,r2)" if cover_count==2 and the two terms are r1,r2
    
    # Build: pair_unique[(r1,r2)] = set of entries uniquely covered by {r1,r2}
    pair_unique = {}
    for e_idx in range(num_entries):
        terms = coverage[e_idx]
        if len(terms) == 2:
            r1, r2 = sorted(terms)
            key = (r1, r2)
            if key not in pair_unique:
                pair_unique[key] = set()
            pair_unique[key].add(e_idx)
    
    # Now for each pair (r1, r2), the unique entries are those in pair_unique[(r1,r2)]
    # Plus any entry covered by exactly 1 term which is either r1 or r2
    # Precompute single-cover entries per term
    single_cover = {r: set() for r in range(55)}
    for e_idx in range(num_entries):
        terms = coverage[e_idx]
        if len(terms) == 1:
            single_cover[terms[0]].add(e_idx)
    
    # For pair (r1, r2): unique = pair_unique.get((r1,r2), set()) | single_cover[r1] | single_cover[r2]
    # But we need to be careful: entries in single_cover[r1] might also be in pair_unique for another pair
    # Actually no: if an entry is in single_cover[r1], it's covered only by r1, so it's not in any pair_unique
    
    # Let's just iterate over all pairs and check feasibility
    num_terms = 55
    tried = 0
    for r1 in range(num_terms):
        if _check_deadline():
            return None
        for r2 in range(r1 + 1, num_terms):
            if _check_deadline():
                return None
            tried += 1
            
            # Get unique entries for this pair
            key = (r1, r2)
            unique = set(pair_unique.get(key, set()))
            unique |= single_cover[r1]
            unique |= single_cover[r2]
            
            # The new term must cover all unique entries
            # Also, the remaining 53 terms must cover everything else
            # Check: is there any entry NOT covered by the remaining 53 terms AND not in unique?
            # An entry is not covered by remaining 53 terms if it's only covered by r1 or r2
            # That's exactly: entries where all covering terms are in {r1, r2}
            # = entries where coverage subset of {r1, r2}
            # = single_cover[r1] | single_cover[r2] | pair_unique.get(key, set())
            # Which is exactly what we computed as 'unique'
            
            # So the new term covers 'unique', and the other 53 terms cover the rest
            # Total coverage = unique | (union of other 53 terms)
            # The other 53 terms cover all entries except those only covered by r1 or r2
            # So total = unique | (all - unique) = all. Good.
            
            # Build new support
            new_support = []
            for r in range(num_terms):
                if r != r1 and r != r2:
                    new_support.append(set(base_support[r]))
            # Convert unique entry indices back to (i,j,k) tuples
            entries_list = _tensor_entries()
            new_term = set(entries_list[e] for e in unique)
            new_support.append(new_term)
            
            if len(new_support) != 54:
                continue
            
            # Verify full coverage
            all_covered = set()
            for s in new_support:
                all_covered |= s
            if len(all_covered) < num_entries:
                continue
            
            result = _solve_support(new_support, 54)
            if result is not None:
                U, V, W = result
                if _verify_np(U, V, W, tol=1e-7):
                    if _verify_exact(U, V, W):
                        return U, V, W
    
    # Fallback: try merging a schoolbook term with a Strassen term
    # The 20 schoolbook terms (indices 35-54) each cover exactly one entry
    # A Strassen term covers 1-2 entries
    # If we merge a schoolbook term with a Strassen term, the new term covers 2-3 entries
    # This might be solvable
    
    # Try: for each schoolbook term s (35-54) and each Strassen term t (0-34),
    # remove both, add one term covering their union
    for s in range(35, 55):
        if _check_deadline():
            return None
        for t in range(0, 35):
            if _check_deadline():
                return None
            # Check if any entry is only covered by s or t
            unique = set()
            for e_idx in range(num_entries):
                terms = coverage[e_idx]
                if len(terms) == 1 and terms[0] in (s, t):
                    unique.add(e_idx)
                elif len(terms) == 2 and set(terms) == {s, t}:
                    unique.add(e_idx)
            
            entries_list = _tensor_entries()
            new_term = set(entries_list[e] for e in unique)
            # Also include entries from base_support[s] and base_support[t] that are shared
            # Actually, the new term should cover all entries that were covered by s or t
            # and are not covered by any other term
            # That's what 'unique' is
            
            new_support = []
            for r in range(55):
                if r != s and r != t:
                    new_support.append(set(base_support[r]))
            new_support.append(new_term)
            
            if len(new_support) != 54:
                continue
            
            all_covered = set()
            for ss in new_support:
                all_covered |= ss
            if len(all_covered) < num_entries:
                continue
            
            result = _solve_support(new_support, 54)
            if result is not None:
                U, V, W = result
                if _verify_np(U, V, W, tol=1e-7):
                    if _verify_exact(U, V, W):
                        return U, V, W
    
    return None

def construct():
    U55, V55, W55 = _build_strassen_rank55()
    
    result54 = _search_rank54()
    if result54 is not None:
        U, V, W = result54
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U55.tolist(), "V": V55.tolist(), "W": W55.tolist()}
