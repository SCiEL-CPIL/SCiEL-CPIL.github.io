import time
import random
import math
import sys

DEADLINE = time.time() + 145.0

class IntervalSet:
    """
    Efficient 1D interval set for overlap detection.
    Maintains a sorted list of non-overlapping intervals.
    """
    def __init__(self):
        self.intervals = []

    def add(self, start, end):
        """Add interval [start, end) and return False if it overlaps with existing ones."""
        new_start, new_end = start, end
        i = 0
        n = len(self.intervals)
        
        # Find insertion point and check for overlaps
        # We merge overlapping intervals
        while i < n and self.intervals[i][1] <= new_start:
            i += 1
        
        # Check for overlap with previous interval
        if i > 0 and self.intervals[i-1][1] > new_start:
            return False
        
        # Check for overlap with current and subsequent intervals
        # If there is an overlap, return False
        if i < n and self.intervals[i][0] < new_end:
            return False
            
        # No overlap, insert
        self.intervals.insert(i, (new_start, new_end))
        return True

    def try_add(self, start, end):
        """
        Try to add interval. If it fits (no overlap), add it and return True.
        Otherwise, return False without modifying the set.
        """
        # This is a simplified check. For backtracking, we need to be able to undo.
        # A better approach for backtracking is to keep a stack of changes or copy the list.
        # Given 40 boxes, copying a list of ~40 intervals is cheap.
        # Let's implement a pure function check first, then commit.
        # Actually, for CSP, we usually maintain state.
        # Let's use a list of intervals and a helper to check fit.
        pass

def _check_fit_1d(intervals, start, end):
    """Check if [start, end) fits in the set of non-overlapping intervals."""
    for s, e in intervals:
        if start < e and s < end:
            return False
    return True

def _add_interval(intervals, start, end):
    """Add interval to the list (assumes fit is already checked)."""
    intervals.append((start, end))
    # Optional: sort and merge to keep it canonical, but for fit check, unsorted list of disjoint intervals works.
    # However, if we don't merge, we might have many small intervals.
    # For 40 boxes, O(N) check is fine.

def _pack_csp(boxes, container, order, rng):
    """
    Backtracking CSP solver.
    bins: list of dicts, each dict has 'x', 'y', 'z' lists of intervals, and 'vol'.
    """
    W, H, D = container
    n = len(boxes)
    
    # Initialize bins
    # We will try to pack into a fixed number of bins? No, we add bins as needed.
    # To optimize score (minimize B), we should try to fit into existing bins first.
    # We'll maintain a list of bins.
    
    # State:
    # bins: list of (x_intervals, y_intervals, z_intervals, used_volume)
    # Actually, 3D overlap is not just 1D independent.
    # Wait, the problem is 3D. Overlap in 3D requires overlap in ALL 3 axes.
    # Interval arithmetic per axis is only sufficient if we are packing 1D.
    # For 3D, we need to check if the 3D box overlaps with any existing 3D box.
    # The "interval arithmetic" hint in the prompt might imply a specific structure or I should use it for pruning in a different way.
    # Standard 3D bin packing CSP:
    # A box overlaps if: x_overlap AND y_overlap AND z_overlap.
    # We can use interval sets per axis to quickly reject candidates that don't overlap in at least one axis?
    # No, we need to ensure NO overlap.
    # So, for a new box (x1, y1, z1, w, h, d) to be valid:
    # For every existing box (x2, y2, z2, w2, h2, d2):
    #   NOT (x_overlap AND y_overlap AND z_overlap)
    # This is hard to prune with simple 1D intervals unless we build a spatial index.
    
    # Given the time limit and 40 boxes, a simple O(N^2) check per placement is 1600 ops.
    # 40 boxes * 40 positions * 1600 checks = 256,000 ops per full path.
    # With backtracking, this might be too slow if the search space is large.
    # However, with good heuristics (MRV, Value Ordering), it can be fast.
    
    # Let's stick to the O(N) overlap check per candidate position, but generate candidate positions smartly.
    # Candidate positions:
    # 1. Origin (0,0,0)
    # 2. Faces of existing boxes (x2+w2, y2, z2), etc.
    # This is the "Anchor" approach again, but now used in a CSP with backtracking.
    
    # To make it different from the greedy program:
    # 1. We use backtracking (if a box doesn't fit, we try other positions or other bins).
    # 2. We use a stronger ordering (e.g., Largest Volume First).
    # 3. We try to minimize the number of bins by opening new bins only when necessary.
    
    # State representation for backtracking:
    # We need to undo placements.
    # We'll use a recursive function that takes the current state (list of bins, each bin is a list of boxes)
    # and the index of the next box to place.
    
    # To speed up, we'll precompute anchors for each bin? No, anchors change as boxes are added.
    
    # Optimization:
    # Instead of generating all anchors every time, we can just check a limited set of "promising" positions.
    # But for correctness in backtracking, we need to explore enough to find a solution.
    # Since we just need ONE valid packing (and the best one by score), and 40 boxes is small,
    # we can use a greedy approach within the CSP nodes? No, that defeats the purpose.
    
    # Let's try a different angle: **Iterative Deepening A* (IDA*)** or simply **Backtracking with Heuristics**.
    
    # Heuristic for Value Ordering (which position to try first):
    # - Bottom-Left-Front (minimize x, then y, then z)
    # - Closest to existing boxes (maximize contact)
    
    # Heuristic for Variable Ordering (which box to place next):
    # - Fixed by the `order` passed in (e.g., Volume Descending).
    
    # Let's implement a backtracking solver that:
    # 1. Takes a specific order.
    # 2. Tries to place each box in existing bins (trying positions) or a new bin.
    # 3. If it gets stuck, it backtracks.
    # 4. We limit the depth of backtracking or the number of positions tried per node to stay within time.
    
    # Actually, with 40 boxes, full backtracking might explode.
    # We should use **Limited Backtracking** or **Restarting**.
    
    # Strategy:
    # 1. Generate a few good orderings (Vol Desc, Max Dim Desc).
    # 2. For each ordering, run a backtracking solver with a node limit.
    # 3. If it finds a solution, record it.
    # 4. Compare scores.
    
    # To make it efficient, the "fit" check must be fast.
    
    def get_anchors(placements, W, H, D):
        anchors = [(0, 0, 0)]
        for (px, py, pz, pw, ph, pd) in placements:
            # Primary faces
            if px + pw < W: anchors.append((px + pw, py, pz))
            if py + ph < H: anchors.append((px, py + ph, pz))
            if pz + pd < D: anchors.append((px, py, pz + pd))
            # Secondary combinations (corners/edges)
            if px + pw < W and py + ph < H: anchors.append((px + pw, py + ph, pz))
            if px + pw < W and pz + pd < D: anchors.append((px + pw, py, pz + pd))
            if py + ph < H and pz + pd < D: anchors.append((px, py + ph, pz + pd))
            if px + pw < W and py + ph < H and pz + pd < D: anchors.append((px + pw, py + ph, pz + pd))
        
        # Remove duplicates
        return list(set(anchors))
    
    def fits(placements, cx, cy, cz, w, h, d, W, H, D):
        if cx < 0 or cy < 0 or cz < 0:
            return False
        if cx + w > W or cy + h > H or cz + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in placements:
            if (cx < px + pw and px < cx + w and
                cy < py + ph and py < cy + h and
                cz < pz + pd and pz < cz + d):
                return False
        return True
    
    def solve(idx, bins):
        """
        Recursively place box at index `idx`.
        bins: list of lists of (x, y, z, w, h, d)
        Returns True if success, False if fail.
        """
        if time.time() > DEADLINE:
            return False
            
        if idx == n:
            return True
            
        w, h, d = boxes[order[idx]]
        
        # Try to place in existing bins
        # Order bins by "fullness" or "closeness" to encourage filling?
        # Let's try bins in the order they were created, or sorted by remaining volume?
        # Sorting bins every step is expensive. Let's just iterate.
        
        for bi in range(len(bins)):
            placements = bins[bi]
            anchors = get_anchors(placements, W, H, D)
            
            # Heuristic: Sort anchors to prefer lower z, then lower y, then lower x
            # This encourages compact packing.
            anchors.sort(key=lambda p: (p[2], p[1], p[0]))
            
            for (cx, cy, cz) in anchors:
                if fits(placements, cx, cy, cz, w, h, d, W, H, D):
                    # Place box
                    bins[bi].append((cx, cy, cz, w, h, d))
                    
                    if solve(idx + 1, bins):
                        return True
                    else:
                        # Backtrack
                        bins[bi].pop()
                        
            # If no anchor worked in this bin, try the next bin
            # (Implicitly handled by the loop)
            
        # If no existing bin works, try a new bin
        # We should limit the number of new bins to avoid explosion?
        # The score penalizes B, so we want to minimize bins.
        # If we open a new bin, we place at (0,0,0).
        
        # Heuristic: Only open a new bin if the current box is large enough to justify it?
        # Or just allow it, but the backtracking will try to avoid it if possible by finding space in existing bins.
        # However, if we always try existing bins first, we might fill them up and force a new bin later.
        
        # Let's allow opening a new bin.
        new_bin = [(0, 0, 0, w, h, d)]
        bins.append(new_bin)
        
        if solve(idx + 1, bins):
            return True
        else:
            bins.pop()
            return False
            
    # Run the solver
    initial_bins = []
    # To help the solver, we can pre-place the first box?
    # Or just let it place box 0 in bin 0.
    
    # We need to be careful with the number of bins.
    # If the solver opens too many bins, the score will be bad.
    # We can add a constraint: max_bins.
    # But we don't know the optimal number of bins.
    # We can iterate with increasing max_bins?
    # Or just let it find a solution and check the score.
    
    # To prevent infinite loops or too many bins, we can check if len(bins) > some limit.
    # Let's set a soft limit: if len(bins) > n, return False (shouldn't happen).
    
    # One issue: The solver might find a solution with many bins, when a better one with fewer exists.
    # Backtracking will explore other options if the current path fails, but if it succeeds with 5 bins, it stops.
    # It doesn't guarantee the MINIMUM number of bins.
    # To get a better score, we should try to find a solution with fewer bins.
    
    # Strategy:
    # 1. Run solver with a limit on the number of bins (e.g., start with 1 bin, increase if fails).
    # 2. This ensures we find the packing with the minimum number of bins (or close to it).
    
    def solve_with_limit(idx, bins, max_bins):
        if time.time() > DEADLINE:
            return False
        if idx == n:
            return True
        if len(bins) > max_bins:
            return False
            
        w, h, d = boxes[order[idx]]
        
        for bi in range(len(bins)):
            placements = bins[bi]
            anchors = get_anchors(placements, W, H, D)
            anchors.sort(key=lambda p: (p[2], p[1], p[0]))
            
            for (cx, cy, cz) in anchors:
                if fits(placements, cx, cy, cz, w, h, d, W, H, D):
                    bins[bi].append((cx, cy, cz, w, h, d))
                    if solve_with_limit(idx + 1, bins, max_bins):
                        return True
                    bins[bi].pop()
                    
        if len(bins) < max_bins:
            new_bin = [(0, 0, 0, w, h, d)]
            bins.append(new_bin)
            if solve_with_limit(idx + 1, bins, max_bins):
                return True
            bins.pop()
            
        return False
        
    # Try to find the minimum number of bins
    # Start with 1 bin, increase until a solution is found
    # Note: This might be slow if the minimum number of bins is high.
    # But for 40 boxes, the number of bins is likely small (e.g., 2-5).
    
    best_solution = None
    for max_bins in range(1, n + 1):
        if time.time() > DEADLINE:
            break
        bins = []
        if solve_with_limit(0, bins, max_bins):
            # Found a solution with max_bins bins (or fewer, but we stopped at max_bins)
            # Actually, solve_with_limit allows UP TO max_bins.
            # It might use fewer.
            best_solution = bins
            break
            
    return best_solution

def _score_solution(bins, container):
    W, H, D = container
    C = W * H * D
    B = len(bins)
    if B == 0:
        return 0.0
    last_bin = bins[-1]
    if not last_bin:
        return B
    last_vol = sum(w * h * d for (_, _, _, w, h, d) in last_bin)
    fill_ratio = last_vol / C if C > 0 else 0.0
    return B + 0.1 * (1.0 - fill_ratio)

def _generate_orderings(boxes, n, rng):
    idx = list(range(n))
    # 1. Volume Descending
    yield sorted(idx, key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    # 2. Max Dimension Descending
    yield sorted(idx, key=lambda i: max(boxes[i]), reverse=True)
    # 3. Sum of Dimensions Descending
    yield sorted(idx, key=lambda i: sum(boxes[i]), reverse=True)
    # 4. Random
    for _ in range(3):
        s = idx[:]
        rng.shuffle(s)
        yield s

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    best_bins = None
    best_score = float('inf')
    t_start = time.time()
    rng = random.Random(42)
    
    # Try different orderings with the CSP solver
    for order in _generate_orderings(boxes, n, rng):
        if time.time() > DEADLINE:
            break
            
        bins = _pack_csp(boxes, container, order, rng)
        
        if bins is not None:
            s = _score_solution(bins, container)
            if s < best_score:
                best_score = s
                best_bins = bins
                # If we find a perfect score (B=1, fill=1), we can stop?
                # Score = 1 + 0.1*(1-1) = 1.0.
                # If best_score <= 1.0: break # Can't get better than 1.0 if B>=1
        
    if best_bins is None:
        # Fallback: Simple greedy
        best_bins = [[(0, 0, 0, w, h, d)] for w, h, d in boxes]
        
    # Construct final placements
    # Map indices
    # The _pack_csp function uses `order` to pick boxes, but `bins` contains the geometry.
    # We need to know WHICH box (original index) is in which bin.
    # My _pack_csp implementation above didn't track indices!
    # I need to modify _pack_csp to track indices.
    
    # Let's re-implement _pack_csp to return (bins, index_map)
    # Or, since we have the `order`, we can reconstruct?
    # No, the backtracking changes the order of placement.
    # We must track indices in the bins.
    
    # Let's rewrite _pack_csp to store (idx, x, y, z, w, h, d) in bins.
    
    # I will create a new version of the solver that tracks indices.
    
    def _pack_csp_tracked(boxes, container, order, rng):
        W, H, D = container
        n = len(boxes)
        
        def get_anchors(placements, W, H, D):
            # placements is list of (idx, x, y, z, w, h, d)
            anchors = [(0, 0, 0)]
            for (idx, px, py, pz, pw, ph, pd) in placements:
                if px + pw < W: anchors.append((px + pw, py, pz))
                if py + ph < H: anchors.append((px, py + ph, pz))
                if pz + pd < D: anchors.append((px, py, pz + pd))
                if px + pw < W and py + ph < H: anchors.append((px + pw, py + ph, pz))
                if px + pw < W and pz + pd < D: anchors.append((px + pw, py, pz + pd))
                if py + ph < H and pz + pd < D: anchors.append((px, py + ph, pz + pd))
                if px + pw < W and py + ph < H and pz + pd < D: anchors.append((px + pw, py + ph, pz + pd))
            return list(set(anchors))
        
        def fits(placements, cx, cy, cz, w, h, d, W, H, D):
            if cx < 0 or cy < 0 or cz < 0:
                return False
            if cx + w > W or cy + h > H or cz + d > D:
                return False
            for (idx, px, py, pz, pw, ph, pd) in placements:
                if (cx < px + pw and px < cx + w and
                    cy < py + ph and py < cy + h and
                    cz < pz + pd and pz < cz + d):
                    return False
            return True
        
        def solve_with_limit(idx, bins, max_bins):
            if time.time() > DEADLINE:
                return False
            if idx == n:
                return True
            if len(bins) > max_bins:
                return False
                
            box_idx = order[idx]
            w, h, d = boxes[box_idx]
            
            for bi in range(len(bins)):
                placements = bins[bi]
                anchors = get_anchors(placements, W, H, D)
                anchors.sort(key=lambda p: (p[2], p[1], p[0]))
                
                for (cx, cy, cz) in anchors:
                    if fits(placements, cx, cy, cz, w, h, d, W, H, D):
                        bins[bi].append((box_idx, cx, cy, cz, w, h, d))
                        
                        if solve_with_limit(idx + 1, bins, max_bins):
                            return True
                        else:
                            bins[bi].pop()
                            
            if len(bins) < max_bins:
                new_bin = [(box_idx, 0, 0, 0, w, h, d)]
                bins.append(new_bin)
                
                if solve_with_limit(idx + 1, bins, max_bins):
                    return True
                else:
                    bins.pop()
                    
            return False
            
        best_solution = None
        for max_bins in range(1, n + 1):
            if time.time() > DEADLINE:
                break
            bins = []
            if solve_with_limit(0, bins, max_bins):
                best_solution = bins
                break
                
        return best_solution
        
    # Re-run the search with the tracked version
    best_bins = None
    best_score = float('inf')
    
    for order in _generate_orderings(boxes, n, rng):
        if time.time() > DEADLINE:
            break
            
        bins = _pack_csp_tracked(boxes, container, order, rng)
        
        if bins is not None:
            s = _score_solution(bins, container)
            if s < best_score:
                best_score = s
                best_bins = bins

    if best_bins is None:
        # Fallback
        best_bins = [[(i, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])] for i in range(n)]
        
    # Flatten to final output
    final_placement_map = {}
    for bi, bin_content in enumerate(best_bins):
        for item in bin_content:
            idx, x, y, z, w, h, d = item
            final_placement_map[idx] = (bi, x, y, z, w, h, d)
            
    result = []
    for i in range(n):
        if i in final_placement_map:
            result.append(final_placement_map[i])
        else:
            result.append((999, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
            
    return result

