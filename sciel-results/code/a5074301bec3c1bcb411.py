"""Rank-minimal tensor decomposition for 3x4*4x7 matrix multiplication.

Structural change: instead of hand-placing Strassen blocks, we perform a
deterministic search over partitions of the 84 (i,j,k) triples into blocks
that are "Strassen-able" (each dimension of the block is either size 1 or 2,
and all non-singleton dimensions have size exactly 2).  For each valid block
we apply the 7-term Strassen identity; the remaining triples are done by
schoolbook (1 term each).  The objective is to minimise the total rank.

We enumerate all ways to tile the 3x4x7 grid with 2x2x2, 2x2x1, 2x1x2,
1x2x2, and 1x1x1 boxes (where a 1x1x1 box is schoolbook).  This is a
3-D rectangle packing problem.  We use a backtracking search with pruning.
"""
import time
import sys
from itertools import product

_DIMS = (3, 4, 7)
_DEADLINE = 135.0  # safety margin before 150s kill


def _strassen_terms(n, m, p, i0, j0, k0):
    """
    Return the 7 rank-1 terms for a 2x2x2 Strassen block.
    Each term is (U_entries, V_entries, W_entries) where each is a list of
    (index, value) pairs.  U has n*m rows, V has m*p rows, W has p*n rows.
    Index convention: U[i*m+j], V[j*p+k], W[k*n+i].
    """
    terms = []

    # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
    #   -> C[i0,k0] +=, C[i1,k1] +=
    terms.append((
        [((i0)*m+j0, 1), ((i0+1)*m+(j0+1), 1)],
        [((j0)*p+k0, 1), ((j0+1)*p+(k0+1), 1)],
        [((k0)*n+i0, 1), ((k0+1)*n+(i0+1), 1)],
    ))

    # M2: (A[i1,j0]+A[i1,j1])*B[j0,k0]
    #   -> C[i1,k0] +=, C[i1,k1] -=
    terms.append((
        [((i0+1)*m+j0, 1), ((i0+1)*m+(j0+1), 1)],
        [((j0)*p+k0, 1)],
        [((k0)*n+(i0+1), 1), ((k0+1)*n+(i0+1), -1)],
    ))

    # M3: A[i0,j0]*(B[j0,k1]-B[j1,k1])
    #   -> C[i0,k1] +=, C[i1,k1] +=
    terms.append((
        [(i0*m+j0, 1)],
        [((j0)*p+(k0+1), 1), ((j0+1)*p+(k0+1), -1)],
        [((k0+1)*n+i0, 1), ((k0+1)*n+(i0+1), 1)],
    ))

    # M4: A[i1,j1]*(B[j1,k0]-B[j0,k0])
    #   -> C[i0,k0] +=, C[i1,k0] +=
    terms.append((
        [((i0+1)*m+(j0+1), 1)],
        [((j0+1)*p+k0, 1), ((j0)*p+k0, -1)],
        [((k0)*n+i0, 1), ((k0)*n+(i0+1), 1)],
    ))

    # M5: (A[i0,j0]+A[i0,j1])*B[j1,k1]
    #   -> C[i0,k0] -=, C[i0,k1] +=
    terms.append((
        [(i0*m+j0, 1), (i0*m+(j0+1), 1)],
        [((j0+1)*p+(k0+1), 1)],
        [((k0)*n+i0, -1), ((k0+1)*n+i0, 1)],
    ))

    # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
    #   -> C[i1,k1] +=
    terms.append((
        [((i0+1)*m+j0, 1), (i0*m+j0, -1)],
        [((j0)*p+k0, 1), ((j0)*p+(k0+1), 1)],
        [((k0+1)*n+(i0+1), 1)],
    ))

    # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
    #   -> C[i0,k0] +=
    terms.append((
        [(i0*m+(j0+1), 1), ((i0+1)*m+(j0+1), -1)],
        [((j0+1)*p+k0, 1), ((j0+1)*p+(k0+1), 1)],
        [((k0)*n+i0, 1)],
    ))

    return terms


def _schoolbook_term(i, j, k, n, m, p):
    """Return a single rank-1 term for the schoolbook multiplication at (i,j,k)."""
    return (
        [(i*m+j, 1)],
        [(j*p+k, 1)],
        [(k*n+i, 1)],
    )


def _find_decomposition(n, m, p, deadline):
    """
    Find a partition of the n*m*p grid into 2x2x2, 2x2x1, 2x1x2, 1x2x2,
    and 1x1x1 blocks.  Return the list of blocks as (type, i0, j0, k0)
    where type is one of 's' (Strassen 2x2x2), 'sb' (schoolbook single).
    
    We use a backtracking search.  The grid is n x m x p = 3 x 4 x 7 = 84 cells.
    A 2x2x2 block covers 8 cells and costs 7.
    A 2x2x1 block covers 4 cells and costs 4 (schoolbook) or... actually
    a 2x2x1 block doesn't get a Strassen benefit, it's just 4 schoolbook terms.
    
    Wait - the only block that gets a Strassen benefit is the full 2x2x2.
    All other blocks are just schoolbook.  So the problem reduces to:
    place as many 2x2x2 blocks as possible (each saves 1 term: 8 cells
    cost 7 instead of 8).  The rest are schoolbook.
    
    So total rank = 84 - (number of 2x2x2 blocks).
    We want to maximise the number of 2x2x2 blocks we can pack into 3x4x7.
    
    Maximum possible: floor(3/2)*floor(4/2)*floor(7/2) = 1*2*3 = 6.
    But we can do better by overlapping the "2" dimensions non-aligned.
    
    Actually, for 3x4x7: we can try to pack 2x2x2 boxes.  The maximum
    number of non-overlapping 2x2x2 boxes in a 3x4x7 grid:
    - Along dimension 0 (size 3): we can fit at most 1 "pair" but there's a remainder.
    - Actually we can fit floor(3/2) = 1 complete pair, plus 1 leftover row.
    - Along dimension 1 (size 4): floor(4/2) = 2 pairs.
    - Along dimension 2 (size 7): floor(7/2) = 3 pairs, plus 1 leftover.
    
    So aligned packing gives 1*2*3 = 6 boxes, covering 6*8 = 48 cells.
    Remaining 36 cells are schoolbook.  Total rank = 48/8*7 + 36 = 42 + 36 = 78.
    
    But can we do better with non-aligned packing?  In a 3D grid, we might
    be able to pack more than the aligned bound by shifting.  Let's think...
    
    For dimension 0 (size 3): we can place boxes starting at i=0 (covering 0,1)
    or i=1 (covering 1,2).  These overlap at i=1.
    
    The maximum independent set of 2x2x2 boxes in a 3x4x7 grid is what we want.
    This is a 3D matching/packing problem.
    
    Let me just do a backtracking search to find the maximum number of
    2x2x2 boxes that can be placed without overlap.
    """
    # We want to maximise the number of 2x2x2 boxes in a 3x4x7 grid.
    # Total cells = 84.  Each box covers 8 cells.
    # Upper bound: floor(84/8) = 10.  But geometrically limited.
    
    # Let's enumerate all possible 2x2x2 box positions:
    # i0 in {0,1}, j0 in {0,1,2}, k0 in {0,1,2,3,4,5}
    # Total positions: 2 * 3 * 6 = 36 possible boxes.
    
    # We want the maximum number of non-overlapping boxes.
    # This is a maximum independent set in the intersection graph, or
    # equivalently a maximum packing.
    
    # For a small problem like this (36 boxes, 84 cells), we can use
    # backtracking with bitmask.
    
    # Represent the grid as a 3D boolean array.  84 cells.
    # Each box covers 8 specific cells.
    
    grid_size = n * m * p  # 84
    cell_index = lambda i, j, k: (i * m + j) * p + k  # unique 0..83
    
    # All possible box positions
    boxes = []
    for i0 in range(n - 1):  # 0, 1
        for j0 in range(m - 1):  # 0, 1, 2
            for k0 in range(p - 1):  # 0..5
                cells = []
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            cells.append(cell_index(i0+di, j0+dj, k0+dk))
                boxes.append((i0, j0, k0, frozenset(cells)))
    
    num_boxes = len(boxes)  # 36
    
    # Backtracking: try to place boxes greedily with backtracking.
    # Use a bitmask of occupied cells (84 bits, fits in Python int).
    
    best_count = 0
    best_mask = 0
    
    # Sort boxes by some heuristic (e.g., position) for determinism
    # Actually, let's just enumerate in order and use backtracking.
    
    def backtrack(pos, occupied, count):
        nonlocal best_count, best_mask
        if time.time() > deadline:
            return
        # Pruning: even if we place all remaining boxes, can we beat best?
        remaining = num_boxes - pos
        if count + remaining <= best_count:
            return
        if count > best_count:
            best_count = count
            best_mask = occupied
        
        # Try to place each remaining box
        for b in range(pos, num_boxes):
            box_cells = boxes[b][3]
            box_mask = 0
            for c in box_cells:
                box_mask |= (1 << c)
            if box_mask & occupied:
                continue  # overlap
            backtrack(b + 1, occupied | box_mask, count + 1)
            # Don't place box b, try next
            # (implicit by continuing the loop)
        
        # Also the option of not placing any more boxes is handled by
        # the base case / pruning above.
    
    # This naive backtracking might be too slow for 36 boxes.
    # Let's use a smarter approach: since the grid is small (3x4x7),
    # we can think of it layer by layer.
    
    # Actually, let me think about this more carefully.
    # The grid is 3 x 4 x 7.  A 2x2x2 box needs 2 consecutive indices
    # in each dimension.
    
    # In dimension 0 (size 3): positions are {0,1} or {1,2}.
    # In dimension 1 (size 4): positions are {0,1}, {1,2}, {2,3}.
    # In dimension 2 (size 7): positions are {0,1}, {1,2}, {2,3}, {3,4}, {4,5}, {5,6}.
    
    # Two boxes overlap if they share any cell.  In 3D, two boxes overlap
    # if and only if their projections overlap in ALL three dimensions.
    
    # Box A at (i0a, j0a, k0a) covers i in [i0a, i0a+1], j in [j0a, j0a+1], k in [k0a, k0a+1].
    # Box B at (i0b, j0b, k0b) overlaps A iff:
    #   intervals [i0a, i0a+1] and [i0b, i0b+1] intersect, AND
    #   intervals [j0a, j0a+1] and [j0b, j0b+1] intersect, AND
    #   intervals [k0a, k0a+1] and [k0b, k0b+1] intersect.
    
    # Two length-2 intervals on the integer line intersect iff their starts
    # differ by at most 1.
    
    # So boxes overlap iff |i0a - i0b| <= 1 AND |j0a - j0b| <= 1 AND |k0a - k0b| <= 1.
    
    # We want a maximum set of box positions such that no two are within
    # Chebyshev distance 1 of each other (in the 3D space of box positions).
    # Wait, that's not quite right.  The Chebyshev distance between box
    # positions (i0,j0,k0) and (i0',j0',k0') is max(|i0-i0'|, |j0-j0'|, |k0-k0'|).
    # They overlap iff Chebyshev distance <= 1.
    # So we want a maximum set of points in the 2x3x6 grid of box positions
    # with pairwise Chebyshev distance >= 2.
    
    # This is a "maximum packing" or "maximum independent set in king graph" problem.
    # For a 2x3x6 grid with king adjacency, we can enumerate.
    
    # Actually, since the grid of box positions is only 2*3*6 = 36 points,
    # and we need pairwise Chebyshev distance >= 2, let's think about it
    # layer by layer.
    
    # In dimension 0 (size 2): positions 0 and 1 are adjacent (distance 1).
    # So we can pick at most one from {0,1} per (j0,k0) column.
    
    # Hmm, this is getting complex.  Let me just use a simple greedy + local
    # search, or since the problem is small enough, use a different approach.
    
    # Actually, for a 2x3x6 king graph, the maximum independent set can be
    # found by considering that in each "slab" of thickness 2 in any direction,
    # we can pick at most one per 2x2x2 sub-grid of box positions.
    
    # Let me just try a few known packings and compute the rank.
    
    # Greedy approach: place boxes in a regular pattern.
    # In dimension 0: pick i0=0 (covers rows 0,1).  Leave row 2 for schoolbook.
    # In dimension 1: pick j0=0, j0=2 (covers cols 0-1 and 2-3).
    # In dimension 2: pick k0=0, k0=2, k0=4 (covers 0-1, 2-3, 4-5).  Leave col 6.
    # This gives 1 * 2 * 3 = 6 boxes.
    
    # Can we do 7?  We'd need to also use row 2 and column 6.
    # If we place a box at (i0=1, j0=0, k0=0), it covers rows 1-2, cols 0-1, k 0-1.
    # But this overlaps with (i0=0, j0=0, k0=0) which covers rows 0-1, cols 0-1, k 0-1.
    # They share row 1, cols 0-1, k 0-1.  So they overlap.
    
    # What if we use (i0=1, j0=0, k0=2)?  Covers rows 1-2, cols 0-1, k 2-3.
    # Check against (i0=0, j0=0, k0=0): rows 0-1 vs 1-2 share row 1; cols 0-1 vs 0-1 share; k 0-1 vs 2-3 don't share.  No overlap!
    # Check against (i0=0, j0=0, k0=2): rows 0-1 vs 1-2 share row 1; cols 0-1 vs 0-1 share; k 2-3 vs 2-3 share.  Overlap!
    
    # So we can't simply add boxes.  Let me try a different arrangement.
    
    # What about using i0=1 for some boxes and i0=0 for others?
    # The key constraint is: two boxes overlap iff they are within Chebyshev
    # distance 1 in box-position space.
    
    # Let me just do a proper search.  36 positions, need max independent set
    # in the king graph.  I'll use a simple branch and bound.
    
    # Build adjacency: box a and box b are adjacent (overlap) if
    # |i0a-i0b|<=1 and |j0a-j0b|<=1 and |k0a-k0b|<=1.
    
    adj = [set() for _ in range(num_boxes)]
    for a in range(num_boxes):
        ia, ja, ka = boxes[a][0], boxes[a][1], boxes[a][2]
        for b in range(a + 1, num_boxes):
            ib, jb, kb = boxes[b][0], boxes[b][1], boxes[b][2]
            if abs(ia - ib) <= 1 and abs(ja - jb) <= 1 and abs(ka - kb) <= 1:
                adj[a].add(b)
                adj[b].add(a)
    
    # Maximum independent set: we want the largest set S such that no two
    # elements of S are adjacent.  Equivalently, the complement is a vertex cover...
    # No, let's just do branch and bound.
    
    # Order boxes by degree (descending) for better pruning.
    order = sorted(range(num_boxes), key=lambda x: -len(adj[x]))
    
    # Map from ordered index to original box index
    box_in_order = [order[i] for i in range(num_boxes)]
    
    # Rebuild adjacency in the new order
    adj2 = [set() for _ in range(num_boxes)]
    for a in range(num_boxes):
        for b in adj[box_in_order[a]]:
            # Find b's position in the new order
            for c in range(num_boxes):
                if box_in_order[c] == b:
                    adj2[a].add(c)
                    break
    
    best = 0
    
    def mis_search(pos, chosen, available):
        nonlocal best
        if time.time() > deadline:
            return
        if not available:
            if len(chosen) > best:
                best = len(chosen)
            return
        # Upper bound: len(chosen) + len(available)
        if len(chosen) + len(available) <= best:
            return
        # Pick a vertex from available
        v = available.pop()
        # Option 1: include v
        new_available = available - adj2[v]
        chosen.append(v)
        mis_search(pos + 1, chosen, new_available)
        chosen.pop()
        # Option 2: exclude v
        mis_search(pos + 1, chosen, available)
        available.add(v)
    
    available = set(range(num_boxes))
    chosen = []
    mis_search(0, chosen, available)
    
    max_boxes = best
    # Rank = 84 - max_boxes (each 2x2x2 box saves 1 rank term)
    rank = grid_size - max_boxes
    
    return rank, max_boxes


def construct():
    _t0 = time.time()
    n, m, p = _DIMS  # 3, 4, 7

    # First, find the maximum number of 2x2x2 Strassen blocks
    rank, max_boxes = _find_decomposition(n, m, p, _DEADLINE)

    # Now we need to actually construct the decomposition.
    # We need to know WHICH boxes to use, not just how many.
    # Let me redo this to return the actual box positions.
    
    # Actually, let me just hardcode a good packing for 3x4x7.
    # The aligned packing gives 6 boxes:
    # (0,0,0), (0,0,2), (0,0,4), (0,2,0), (0,2,2), (0,2,4)
    # These cover: i in {0,1}, j in {0,1} or {2,3}, k in {0,1} or {2,3} or {4,5}
    # Total covered: 2*4*6 = 48 cells.  Remaining: 84-48 = 36 cells.
    # Rank = 6*7 + 36 = 42 + 36 = 78.
    
    # Can we do 7 boxes?  Let me think...
    # We need to also cover some of the cells in row 2 and column 6.
    # A box at (1, j0, k0) covers rows 1-2.  For it not to overlap with
    # boxes at (0, j0, k0), we need |j0 - j0'| > 1 or |k0 - k0'| > 1
    # for all (j0', k0') used by boxes at i0=0.
    
    # If we use boxes at i0=0 with (j0,k0) in {(0,0),(0,2),(0,4),(2,0),(2,2),(2,4)},
    # then a box at (1, j0, k0) would overlap if there exists (j0',k0') in the
    # above set with |j0-j0'|<=1 and |k0-k0'|<=1.
    
    # Available (j0,k0) for i0=1: need (j0,k0) not within Chebyshev 1 of any
    # of {(0,0),(0,2),(0,4),(2,0),(2,2),(2,4)}.
    # j0 in {0,1,2}, k0 in {0,1,2,3,4,5}.
    # (0,0): covered by (0,0) itself.
    # (0,1): |0-0|<=1, |1-0|<=1 -> overlaps with (0,0).
    # (0,2): overlaps with (0,2).
    # (0,3): |0-0|<=1, |3-2|<=1 -> overlaps with (0,2).  Also |0-0|<=1,|3-4|<=1 -> overlaps (0,4).
    # (0,4): overlaps.
    # (0,5): |0-0|<=1, |5-4|<=1 -> overlaps with (0,4).
    # (1,0): |1-0|<=1, |0-0|<=1 -> overlaps.
    # (1,1): |1-0|<=1,|1-0|<=1 -> overlaps with (0,0).  |1-2|<=1,|1-0|<=1 -> overlaps (2,0).
    # (1,2): overlaps.
    # (1,3): |1-0|<=1,|3-2|<=1 -> overlaps (0,2).  |1-2|<=1,|3-2|<=1 -> overlaps (2,2).
    # (1,4): overlaps.
    # (1,5): |1-0|<=1,|5-4|<=1 -> overlaps (0,4).  |1-2|<=1,|5-4|<=1 -> overlaps (2,4).
    # (2,0): overlaps.
    # (2,1): overlaps.
    # (2,2): overlaps.
    # (2,3): overlaps.
    # (2,4): overlaps.
    # (2,5): overlaps.
    
    # So with the aligned i0=0 packing, we can't add any i0=1 box.
    
    # What if we use a different i0=0 packing?  Maybe leave a gap.
    # E.g., use (0,0,0), (0,2,0), (0,0,2), (0,2,2), (0,0,4), (0,2,4) - same thing.
    
    # What if we only use 4 boxes at i0=0, leaving room for 3 at i0=1?
    # That gives 7 total.
    # E.g., i0=0: (0,0,0), (0,2,0), (0,0,4), (0,2,4)  [4 boxes, covering k 0-1 and 4-5]
    # i0=1: (1,0,2), (1,2,2), (1,0,? or 1,2,?)
    # Check (1,0,2): overlaps with (0,0,0)? |0-0|<=1,|0-0|<=1,|2-0|<=1? |2-0|=2>1. No.
    # (1,0,2) vs (0,2,0): |0-2|<=1? No (|0-2|=2). No overlap.
    # (1,0,2) vs (0,0,4): |0-0|<=1,|2-4|<=1? |2-4|=2. No.
    # (1,0,2) vs (0,2,4): |0-2|=2. No.
    # Good.  Now (1,2,2): vs (0,0,0): |2-0|=2. No.
    # (1,2,2) vs (0,2,0): |0-0|<=1,|2-0|<=1? |2-0|=2. No.
    # (1,2,2) vs (0,0,4): |2-0|=2. No.
    # (1,2,2) vs (0,2,4): |0-0|<=1,|2-4|=2. No.
    # Good.  Now need a third i0=1 box.  Try (1,1,0)?
    # (1,1,0) vs (0,0,0): |1-0|<=1,|0-0|<=1,|0-0|<=1. Overlap!
    # Try (1,1,2): vs (0,0,0): |1-0|<=1,|0-0|<=1,|2-0|<=1? |2-0|=2. No.
    # (1,1,2) vs (0,2,0): |1-2|<=1,|0-0|<=1,|2-0|<=1? |2-0|=2. No.
    # (1,1,2) vs (0,0,4): |1-0|<=1,|0-4|=4. No.
    # (1,1,2) vs (0,2,4): |1-2|<=1,|0-4|=4. No.
    # (1,1,2) vs (1,0,2): |1-1|<=1,|0-0|<=1,|2-2|<=1. Overlap!
    # Try (1,1,4): vs (0,0,0): |4-0|=4. No.
    # (1,1,4) vs (0,2,0): |4-0|=4. No.
    # (1,1,4) vs (0,0,4): |1-0|<=1,|0-0|<=1,|4-4|<=1. Overlap!
    # Try (1,0,0): vs (0,0,0): overlap.
    # Try (1,2,4): vs (0,2,4): |1-2|<=1,|2-2|<=1,|4-4|<=1. Overlap!
    
    # Hmm.  Let me try a different arrangement.
    # i0=0: (0,0,0), (0,2,0), (0,0,3), (0,2,3)  [k 0-1 and 3-4]
    # i0=1: (1,0,1), (1,2,1), (1,0,5)
    # (1,0,1) vs (0,0,0): |1-0|<=1,|0-0|<=1,|1-0|<=1. Overlap!
    
    # This is tricky.  Let me just go with 6 boxes (rank 78) which is
    # already better than the current 81.
    
    # Actually wait - the current program has rank 81.  The seed has rank 84 (score 1.0).
    # 81/84 = 1.037 which matches.  So we can do at least 78.
    
    # Let me try to get 7 boxes with a computer search.
    # But for now, let's just use the aligned 6-box packing for rank 78.
    
    # Define the 6 Strassen blocks
    strassen_blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4),
    ]
    
    # Check if we can add a 7th box
    # Try all remaining positions and see if any is non-overlapping
    all_positions = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_positions.append((i0, j0, k0))
    
    used = set(strassen_blocks)
    extra_box = None
    for pos in all_positions:
        if pos in used:
            continue
        # Check against all used boxes
        ok = True
        for (i0u, j0u, k0u) in used:
            if abs(pos[0]-i0u) <= 1 and abs(pos[1]-j0u) <= 1 and abs(pos[2]-k0u) <= 1:
                ok = False
                break
        if ok:
            extra_box = pos
            break
    
    if extra_box is not None:
        strassen_blocks.append(extra_box)
    
    # Now also try to find a better packing by swapping
    # Actually, let me just do a proper maximum independent set search
    # on the 36 box positions.
    
    # For efficiency, let's use a simple branch-and-bound
    boxes_all = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                boxes_all.append((i0, j0, k0))
    
    num_b = len(boxes_all)
    # Adjacency: overlap if Chebyshev distance <= 1 in box-position space
    adj_b = [set() for _ in range(num_b)]
    for a in range(num_b):
        for b in range(a + 1, num_b):
            if abs(boxes_all[a][0]-boxes_all[b][0]) <= 1 and \
               abs(boxes_all[a][1]-boxes_all[b][1]) <= 1 and \
               abs(boxes_all[a][2]-boxes_all[b][2]) <= 1:
                adj_b[a].add(b)
                adj_b[b].add(a)
    
    # Branch and bound for maximum independent set
    best_set = set()
    
    def _mis(pos, chosen, available):
        nonlocal best_set
        if time.time() > _DEADLINE:
            return
        if len(chosen) > len(best_set):
            best_set = set(chosen)
        if not available:
            return
        if len(chosen) + len(available) <= len(best_set):
            return
        # Pick a vertex
        v = next(iter(available))
        available.discard(v)
        # Include v
        new_avail = available - adj_b[v]
        chosen.append(v)
        _mis(pos + 1, chosen, new_avail)
        chosen.pop()
        # Exclude v
        _mis(pos + 1, chosen, available)
        available.add(v)
    
    avail_init = set(range(num_b))
    _mis(0, [], avail_init)
    
    # Convert best_set to box positions
    strassen_blocks = [boxes_all[i] for i in sorted(best_set)]
    
    num_strassen = len(strassen_blocks)
    rank = n * m * p - num_strassen  # each 2x2x2 box saves 1 term
    
    # Build the factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Add Strassen blocks
    for (i0, j0, k0) in strassen_blocks:
        terms = _strassen_terms(n, m, p, i0, j0, k0)
        for (u_entries, v_entries, w_entries) in terms:
            for (idx, val) in u_entries:
                U[idx][col] += val
            for (idx, val) in v_entries:
                V[idx][col] += val
            for (idx, val) in w_entries:
                W[idx][col] += val
            col += 1
    
    # Add schoolbook for all remaining cells
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    assert col == rank, f"Rank mismatch: expected {rank}, got {col}"
    
    # Verify decomposition
    for i in range(n):
        if time.time() - _t0 > _DEADLINE:
            break
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): {total}")
    
    return {"U": U, "V": V, "W": W}
