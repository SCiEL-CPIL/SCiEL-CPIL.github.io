"""ILP-based covering design C(v,k,t) using scipy.optimize.milp (HiGHS).

Strategy:
  - Enumerate all candidate k-subsets (blocks).
  - For each, compute which t-subsets (triples) it covers.
  - Solve a binary set-cover ILP: minimize sum(x) s.t. every triple is covered.
  - If the solver returns an infeasible/incomplete solution, fall back to a
    vectorized greedy repair on the still-uncovered triples.
"""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def construct(v, k, t):
    start = time.time()
    deadline = start + 135.0  # 15s safety margin

    def elapsed():
        return time.time() - start

    # ---- Enumerate candidate blocks and their covered triples ----
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)

    # Precompute triple index lookup (for fast membership checks)
    triple_to_idx = {tr: i for i, tr in enumerate(triples)}

    # Build constraint matrix A (n_triples x n_blocks) with vectorized ops
    # Each block covers C(k, t) triples.
    A = np.zeros((n_triples, n_blocks), dtype=np.int8)
    for j, blk in enumerate(blocks):
        for tr in itertools.combinations(blk, t):
            A[triple_to_idx[tr], j] = 1

    # ---- Solve ILP: minimize sum(x_j) s.t. A @ x >= 1, x in {0,1} ----
    c = np.ones(n_blocks)
    constraints = [LinearConstraint(A, 1, np.inf)]
    integrality = np.ones(n_blocks)
    bounds = Bounds(0, 1)

    best_blocks = None
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality,
                   bounds=bounds, options={"time_limit": 115, "mip_rel_gap": 0.0})
        if res.status == 0:  # optimal
            x = res.x
            best_blocks = [list(blocks[j]) for j in range(n_blocks) if x[j] > 0.5]
        elif res.status in (1, 2, 3):  # infeasible / unbounded / iteration limit
            # Use the best incumbent if available
            if res.x is not None:
                x = res.x
                # Check if incumbent covers everything; if not, repair
                covered = np.zeros(n_triples, dtype=bool)
                for j in range(n_blocks):
                    if x[j] > 0.5:
                        covered |= (A[:, j] > 0)
                if covered.all():
                    best_blocks = [list(blocks[j]) for j in range(n_blocks) if x[j] > 0.5]
    except Exception:
        pass

    # ---- Fallback / repair: vectorized greedy on uncovered triples ----
    if best_blocks is None:
        # Start with a quick greedy seed
        uncovered_mask = np.ones(n_triples, dtype=bool)
        greedy_blocks = []
        while uncovered_mask.any() and elapsed() < deadline - 5:
            # For each candidate block, count how many uncovered triples it covers
            gains = (A[uncovered_mask] > 0).sum(axis=0)
            # Pick the block with max gain (tie-break: first)
            j = int(np.argmax(gains))
            if gains[j] == 0:
                break
            greedy_blocks.append(list(blocks[j]))
            uncovered_mask &= ~(A[:, j] > 0)

        best_blocks = greedy_blocks
    else:
        # Verify and repair if needed
        covered = np.zeros(n_triples, dtype=bool)
        for blk in best_blocks:
            for tr in itertools.combinations(blk, t):
                covered[triple_to_idx[tr]] = True
        if not covered.all():
            # Repair uncovered triples with greedy
            uncovered_mask = ~covered
            while uncovered_mask.any() and elapsed() < deadline - 2:
                gains = (A[uncovered_mask] > 0).sum(axis=0)
                j = int(np.argmax(gains))
                if gains[j] == 0:
                    break
                best_blocks.append(list(blocks[j]))
                uncovered_mask &= ~(A[:, j] > 0)

    # ---- Final validation and dedup ----
    seen = set()
    final_blocks = []
    for blk in best_blocks:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            final_blocks.append(list(key))

    # Final safety check: ensure every triple is covered
    # (If not, do a last-resort greedy pass)
    if elapsed() < deadline - 1:
        covered_final = np.zeros(n_triples, dtype=bool)
        for blk in final_blocks:
            for tr in itertools.combinations(blk, t):
                covered_final[triple_to_idx[tr]] = True
        if not covered_final.all():
            uncovered_mask = ~covered_final
            while uncovered_mask.any() and elapsed() < deadline - 0.5:
                gains = (A[uncovered_mask] > 0).sum(axis=0)
                j = int(np.argmax(gains))
                if gains[j] == 0:
                    break
                blk = list(blocks[j])
                key = tuple(sorted(blk))
                if key not in seen:
                    seen.add(key)
                    final_blocks.append(blk)
                uncovered_mask &= ~(A[:, j] > 0)

    return final_blocks
# EVOLVE-BLOCK-END
