"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction
    deadline = time.time() + 80.0
    denoms = list(range(2, 49))
    n = len(denoms)

    def greedy():
        selected, total = [], Fraction(0)
        for i, d in enumerate(denoms):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected

    def exact_sum(indices):
        s = Fraction(0)
        for j in indices:
            s += Fraction(1, denoms[j])
        return s

    try:
        import numpy as np
        vals = [1.0 / d for d in denoms]
        nA = 23
        nB = n - nA
        A = vals[:nA]
        B = vals[nA:]

        sizeA = 1 << nA
        masksA = np.arange(sizeA, dtype=np.int32)
        sumsA = np.zeros(sizeA, dtype=np.float64)
        for i in range(nA):
            sumsA += A[i] * ((masksA >> i) & 1)
        order = np.argsort(sumsA)
        sumsA_sorted = sumsA[order]
        masksA_sorted = masksA[order]

        sizeB = 1 << nB
        masksB = np.arange(sizeB, dtype=np.int32)
        sumsB = np.zeros(sizeB, dtype=np.float64)
        for i in range(nB):
            sumsB += B[i] * ((masksB >> i) & 1)

        targets = 1.0 - sumsB
        idx = np.searchsorted(sumsA_sorted, targets, side='right') - 1
        valid = idx >= 0
        cand_idx = np.where(valid, idx, 0)
        cand_sum = np.where(valid, sumsB + sumsA_sorted[cand_idx], -np.inf)

        K = min(1000, sizeB)
        topB = np.argpartition(cand_sum, -K)[-K:]
        topB = topB[np.argsort(cand_sum[topB])[::-1]]

        best = None
        best_frac = None
        for bi in topB:
            if time.time() > deadline:
                break
            ai = int(cand_idx[bi])
            maskA = int(masksA_sorted[ai])
            maskB = int(masksB[bi])
            indices = [i for i in range(nA) if (maskA >> i) & 1]
            indices += [nA + i for i in range(nB) if (maskB >> i) & 1]
            if not indices:
                continue
            s = exact_sum(indices)
            if s < 1 and (best_frac is None or s > best_frac):
                best_frac = s
                best = indices
        if best is not None:
            # light local exact polish
            improved = True
            while improved and time.time() < deadline:
                improved = False
                cur = set(best)
                cur_sum = best_frac
                # try swaps and adds
                for add in range(n):
                    if add in cur:
                        continue
                    new_sum = cur_sum + Fraction(1, denoms[add])
                    if new_sum < 1 and new_sum > cur_sum:
                        best = list(cur | {add})
                        cur_sum = new_sum
                        improved = True
                        break
                if improved:
                    continue
                for add in range(n):
                    if add in cur:
                        continue
                    for rem in list(cur):
                        new_sum = cur_sum + Fraction(1, denoms[add]) - Fraction(1, denoms[rem])
                        if new_sum < 1 and new_sum > cur_sum:
                            cur = set(best)
                            cur.add(add)
                            cur.discard(rem)
                            best = list(cur)
                            cur_sum = new_sum
                            improved = True
                            break
                    if improved:
                        break
            return best
    except Exception:
        pass
    return greedy()
# EVOLVE-BLOCK-END
