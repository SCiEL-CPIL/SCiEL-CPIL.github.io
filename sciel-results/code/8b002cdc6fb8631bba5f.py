"""Search-based: minimize max row length for residue covering mod 101."""
import time
import sys

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    deadline = 80.0  # safety margin before 90s kill

    P = 101
    D = list(range(1, 11))

    # Precompute modular inverses
    inv = [0] * 11
    for x in D:
        inv[x] = pow(x, -1, P)

    # Enumerate all nonempty subsets, group by size
    # subset_list[k] = list of (residue, frozenset_of_terms) for subsets of size k
    subsets_by_size = {k: [] for k in range(1, 11)}

    for mask in range(1, 1 << 10):
        terms = []
        res = 0
        for x in D:
            if mask & (1 << (x - 1)):
                terms.append(x)
                res = (res + inv[x]) % P
        k = len(terms)
        subsets_by_size[k].append((res, tuple(sorted(terms))))

    # For each k from 1 to 10, check if all 101 residues are covered
    # by subsets of size <= k. Find minimum such k.
    best_k = 10
    for k in range(1, 11):
        covered = set()
        for kk in range(1, k + 1):
            for res, terms in subsets_by_size[kk]:
                covered.add(res)
        if len(covered) == 101:
            best_k = k
            break

    # Now for each residue, pick a subset of size <= best_k
    # To minimize max row length, we want to use smaller subsets preferentially
    # but we need to cover all residues.
    # Strategy: for each residue, collect all subsets of size <= best_k that map to it.
    # Then assign: prefer smaller subsets.
    #
    # Actually, the objective is to minimize the MAX row length.
    # Since we already found the minimum k such that coverage is possible with size <= k,
    # the answer is k. We just need to produce valid rows.
    #
    # For each residue, pick the smallest-size subset available (size <= best_k).

    # Build residue -> list of (size, terms) for all subsets of size <= best_k
    residue_options = {r: [] for r in range(P)}
    for k in range(1, best_k + 1):
        for res, terms in subsets_by_size[k]:
            residue_options[res].append((k, terms))

    # For each residue, pick the option with minimum size
    result = [None] * P
    for r in range(P):
        if not residue_options[r]:
            # Should not happen if best_k is correct, but fallback
            for k in range(best_k + 1, 11):
                for res, terms in subsets_by_size[k]:
                    if res == r:
                        residue_options[r].append((k, terms))
                        break
                if residue_options[r]:
                    break
        # Pick smallest size
        residue_options[r].sort(key=lambda x: x[0])
        result[r] = list(residue_options[r][0][1])

    # Verify
    for r in range(P):
        if not result[r]:
            raise RuntimeError(f"Residue {r} not covered")
        s = sum(inv[x] for x in result[r]) % P
        if s != r:
            raise RuntimeError(f"Residue {r}: got {s}")

    return result

# EVOLVE-BLOCK-END
