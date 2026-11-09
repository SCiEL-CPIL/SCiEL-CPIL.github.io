"""Find a rank-minimal exact tensor decomposition for 2x4 by 4x6 matrix multiplication."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

# Precompute the set of required (i,j,k) triples
def _required_triples():
    n, m, p = DIMENSIONS
    return set(itertools.product(range(n), range(m), range(p)))

# For a given rank r, we need to assign to each column c a set of (i,j,k) triples
# such that:
#   - For each (i,j,k) in the full set, there exists at least one column c where (i,j,k) is in the support
#   - The support of column c must be "separable": i.e., it must be a product set I_c x J_c x K_c
#     (where I_c subset of rows, J_c subset of cols of A, K_c subset of cols of B)
#   - Actually, the support in the tensor is the set of (i,j,k) such that U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] != 0
#
# Key insight: Each column c defines three sets:
#   I_c = {i : exists j,k with U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] != 0}
#   J_c = {j : exists i,k with ...}
#   K_c = {k : exists i,j with ...}
# And the support is exactly I_c x J_c x K_c (since the tensor entry is a product).
#
# So we need to cover all n*m*p = 48 triples with r "boxes" (product sets).
# The minimum number of boxes needed to cover a 2x4x6 grid is what we're looking for.
#
# This is equivalent to: find the minimum r such that the 2x4x6 0-1 tensor can be
# written as sum of r rank-1 0-1 tensors (each being a product set).
#
# For a 2x4x6 grid, the trivial upper bound is 48 (one box per element).
# We can do better by grouping.
#
# Strategy: Use a greedy/set-cover approach or try small ranks and use backtracking.
# Since 2x4x6 is small, we can enumerate possible box structures.
#
# A box is defined by (I, J, K) where I subset {0,1}, J subset {0,1,2,3}, K subset {0,...,5}.
# Total possible boxes: (2^2 - 1) * (2^4 - 1) * (2^6 - 1) = 3 * 15 * 63 = 2835.
# That's manageable.
#
# We want to find the minimum number of boxes whose union covers all 48 points.
# This is a set cover problem. We can use a greedy approach plus local search,
# or since the problem is small, we can try to find optimal solutions for small ranks.

def _all_boxes():
    n, m, p = DIMENSIONS
    boxes = []
    for size_i in range(1, n + 1):
        for I in itertools.combinations(range(n), size_i):
            for size_j in range(1, m + 1):
                for J in itertools.combinations(range(m), size_j):
                    for size_k in range(1, p + 1):
                        for K in itertools.combinations(range(p), size_k):
                            boxes.append((set(I), set(J), set(K)))
    return boxes

def _box_triples(box):
    I, J, K = box
    return set((i, j, k) for i in I for j in J for k in K)

def _build_decomposition(boxes):
    """Build U, V, W from a list of boxes. Returns None if invalid."""
    n, m, p = DIMENSIONS
    rank = len(boxes)
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    for c, box in enumerate(boxes):
        I, J, K = box
        for i in I:
            for j in J:
                U[i * m + j][c] = Fraction(1)
        for j in J:
            for k in K:
                V[j * p + k][c] = Fraction(1)
        for k in K:
            for i in I:
                W[k * n + i][c] = Fraction(1)
    
    # Verify: the sum of rank-1 tensors must equal the all-ones tensor
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] must be 1 for all (i,j,k)
    # Since each box contributes 1 to (i,j,k) if (i,j,k) is in the box,
    # we need each (i,j,k) to be in exactly one box.
    # Wait, that's not right. The sum must be 1, so each (i,j,k) must be covered exactly once.
    # Actually, the entries can be half-integers, so we could have overlapping boxes
    # with fractional weights. But for simplicity, let's first try 0-1 covers.
    
    # Check coverage
    covered = [0] * (n * m * p)
    for c, box in enumerate(boxes):
        I, J, K = box
        for i in I:
            for j in J:
                for k in K:
                    idx = (i * m + j) * p + k
                    covered[idx] += 1
    
    for idx in range(n * m * p):
        if covered[idx] != 1:
            return None
    
    return {"U": U, "V": V, "W": W}

def _try_greedy_set_cover(target, boxes, max_rank):
    """Greedy set cover: repeatedly pick the box covering the most uncovered points."""
    remaining = set(target)
    chosen = []
    
    while remaining and len(chosen) < max_rank:
        best_box = None
        best_count = -1
        for box in boxes:
            triples = _box_triples(box)
            count = len(remaining & triples)
            if count > best_count:
                best_count = count
                best_box = box
        if best_box is None or best_count == 0:
            break
        chosen.append(best_box)
        remaining -= _box_triples(best_box)
    
    if not remaining:
        return chosen
    return None

def _local_search(boxes, target, max_iter=1000):
    """Try to reduce the number of boxes by swapping."""
    current = list(boxes)
    if not current:
        return current
    
    for _ in range(max_iter):
        improved = False
        for idx in range(len(current)):
            # Try removing box idx and see if we can cover the gap with one other box
            removed = current[idx]
            temp = current[:idx] + current[idx+1:]
            # Find uncovered points
            covered = set()
            for box in temp:
                covered |= _box_triples(box)
            uncovered = target - covered
            if not uncovered:
                current = temp
                improved = True
                break
            # Try to find a single box that covers all uncovered
            for box in boxes:
                if box == removed:
                    continue
                if uncovered <= _box_triples(box):
                    current = temp + [box]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    target = _required_triples()
    all_boxes = _all_boxes()
    
    # Try ranks from small to large
    # Theoretical lower bound: need at least max(n, m, p) = 6 boxes? No, that's not right.
    # Actually, for a 2x4x6 grid, we need to cover 48 points.
    # Each box of size a*b*c covers a*b*c points.
    # Minimum rank: we need sum of box sizes >= 48, and each point covered exactly once.
    # Lower bound: ceil(48 / max_box_size) = ceil(48/48) = 1, but that's only if one box covers all.
    # A box covering all 48 points is (I={0,1}, J={0,1,2,3}, K={0,...,5}), which is valid.
    # So rank 1 should work! Let's verify.
    
    # Rank 1: single box covering everything
    for r in range(1, 49):
        if time.time() > deadline:
            break
        
        # For rank 1, just use the full box
        if r == 1:
            box = (set(range(n)), set(range(m)), set(range(p)))
            result = _build_decomposition([box])
            if result is not None:
                # Convert to int for cleanliness
                for mat in [result["U"], result["V"], result["W"]]:
                    for row in mat:
                        for c in range(len(row)):
                            row[c] = int(row[c])
                return result
        
        # For larger ranks, use greedy set cover
        if time.time() > deadline:
            break
        
        chosen = _try_greedy_set_cover(target, all_boxes, r)
        if chosen is not None:
            # Try to improve with local search
            if time.time() < deadline - 20:
                chosen = _local_search(chosen, target, max_iter=200)
            result = _build_decomposition(chosen)
            if result is not None:
                for mat in [result["U"], result["V"], result["W"]]:
                    for row in mat:
                        for c in range(len(row)):
                            row[c] = int(row[c])
                return result
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
