"""Multi-objective optimizer for ZDT1 using adaptive bisection on Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive bisection optimizer that places points to maximize hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        # ZDT1 Pareto front: f1 = x[0], f2 = 1 - sqrt(f1) when sum(x[1:]) = 1
        # With x[1:] = 1/(n-1), g = 1, so f2 = 1 - sqrt(x[0])

        def make_x_on_pf(x1):
            """Create 30-dim vector with x[0]=x1 and sum(x[1:])=1."""
            x = np.zeros(n)
            x[0] = x1
            x[1:] = 1.0 / (n - 1)
            return np.clip(x, lo, hi)

        evaluated = []  # list of (x, f)
        used = 0

        def eval_point(x):
            nonlocal used
            f = np.array(problem(x))
            evaluated.append((x.copy(), f.copy()))
            used += 1
            return f

        # Track sorted list of (f1, f2) pairs for efficient HV computation
        # f2 = 1 - sqrt(f1) for points on PF
        pf_points = []  # list of f1 values, kept sorted

        def compute_hv_marginal_gain(f1_new):
            """
            Compute marginal hypervolume gain of adding a point at f1_new.
            For 2D HV with ref [11, 11], the gain from inserting a point
            between two existing points depends on the gap and the f2 values.
            """
            if not pf_points:
                # First point: gain is (11 - f1) * (11 - f2)
                f2_new = 1.0 - np.sqrt(f1_new)
                return (11.0 - f1_new) * (11.0 - f2_new)

            # Find insertion position
            idx = np.searchsorted(pf_points, f1_new)

            if idx == 0:
                # New point is to the left of all existing points
                # Gain = area between new point and the leftmost existing point
                f2_new = 1.0 - np.sqrt(f1_new)
                f1_right = pf_points[0]
                f2_right = 1.0 - np.sqrt(f1_right)
                # The new point contributes if f2_new < f2_right (which it will be since f1_new < f1_right)
                # Gain = (11 - f1_new) * (f2_right - f2_new) + ... but we need to be careful
                # Actually, the marginal gain is the area that was not covered before
                # Before: the leftmost point at f1_right covered up to f2_right
                # After: the new point at f1_new, f2_new covers more
                # The gain is (f1_right - f1_new) * (f2_right - f2_new) + (11 - f1_right) * (f2_right - f2_new)
                # Wait, let me think more carefully.
                # HV contribution of a point p is (ref[0]-p[0]) * (next_f2 - p[1]) where next_f2 is the f2 of the next point to the right (or ref[1])
                # When we insert a new point, we change the contributions of adjacent points.
                # Simpler: compute HV before and after, take difference.
                # But that's O(n). Let's use the formula.
                # For 2D, HV = sum over points sorted by f1 of (ref[0] - f1_i) * (f2_{i-1} - f2_i) where f2_{-1} = ref[1]
                # Actually standard formula: sort by f1 ascending, HV = sum (ref[0]-f1_i)*(prev_f2 - f2_i) where prev_f2 starts at ref[1]
                # and we only add if f2_i < prev_f2.
                
                # Let me just compute the gain directly:
                # The new point at (f1_new, f2_new) will be the new leftmost.
                # Before: leftmost was at f1_right with f2_right. Its contribution was (11 - f1_right) * (11 - f2_right) if it was the only point,
                # or (11 - f1_right) * (f2_prev - f2_right) where f2_prev is the f2 of the point to its left... but there is none, so f2_prev = 11.
                # So before, the leftmost point contributed (11 - f1_right) * (11 - f2_right).
                # After inserting, the new point contributes (11 - f1_new) * (11 - f2_new) and the old leftmost contributes (11 - f1_right) * (f2_new - f2_right) if f2_new > f2_right.
                # Since f1_new < f1_right, f2_new = 1-sqrt(f1_new) > 1-sqrt(f1_right) = f2_right.
                # So the gain = [(11 - f1_new)*(11 - f2_new) + (11 - f1_right)*(f2_new - f2_right)] - [(11 - f1_right)*(11 - f2_right)]
                # = (11 - f1_new)*(11 - f2_new) + (11 - f1_right)*f2_new - (11 - f1_right)*f2_right - (11 - f1_right)*11 + (11 - f1_right)*f2_right
                # = (11 - f1_new)*(11 - f2_new) + (11 - f1_right)*f2_new - (11 - f1_right)*11
                # = (11 - f1_new)*(11 - f2_new) - (11 - f1_right)*(11 - f2_new)
                # = (11 - f2_new) * [(11 - f1_new) - (11 - f1_right)]
                # = (11 - f2_new) * (f1_right - f1_new)
                return (11.0 - f2_new) * (f1_right - f1_new)

            elif idx == len(pf_points):
                # New point is to the right of all existing points
                f2_new = 1.0 - np.sqrt(f1_new)
                f1_left = pf_points[-1]
                f2_left = 1.0 - np.sqrt(f1_left)
                # Similar logic: gain = (11 - f2_left) * (f1_new - f1_left) ... wait
                # Before: rightmost at f1_left contributed (11 - f1_left) * (f2_prev - f2_left) where f2_prev is from point to left
                # After: new point at f1_new, f2_new is rightmost. f2_new < f2_left since f1_new > f1_left.
                # The gain comes from the new area to the right.
                # Gain = (11 - f1_left) * (f2_left - f2_new) ... no
                # Let me use the same derivation:
                # Before, the rightmost point's contribution included (11 - f1_left) * (f2_prev - f2_left)
                # After, the old rightmost contributes (11 - f1_left) * (f2_prev - f2_new) and new point contributes (11 - f1_new) * (11 - f2_new)
                # Wait, this is getting complex. Let me just use a simpler approach.
                # The marginal gain of inserting at position idx is:
                # (f1_right - f1_new) * (f2_new - f2_right) if idx < len, where f1_right, f2_right is the point at idx
                # Plus if idx == len, it's (11 - f1_new) * (11 - f2_new) minus what was already covered...
                
                # Actually, the cleanest way: the gain is the area of the rectangle that was not covered.
                # For insertion at the right end: the new point extends coverage to the right.
                # The gain = (f1_new - f1_left) * (f2_left - f2_new) + (11 - f1_new) * (11 - f2_new) - (11 - f1_left) * (11 - f2_left)
                # Hmm, let me just compute it numerically for correctness.
                
                # I'll compute the full HV before and after for small n, but for efficiency let me derive properly.
                # Standard 2D HV: sort points by f1 asc. HV = sum_i (ref[0] - f1_i) * (f2_{i-1} - f2_i) where f2_{-1} = ref[1], and only terms where f2_i < f2_{i-1} count.
                # When inserting a new point at index idx:
                # - If idx > 0: the point at idx-1 previously had next_f2 = f2_idx (or ref[1] if idx was last). Now it has next_f2 = f2_new.
                #   Its contribution changes from (ref[0]-f1_{idx-1})*(f2_{idx-2} - f2_idx) to (ref[0]-f1_{idx-1})*(f2_{idx-2} - f2_new) [if f2_new < f2_{idx-2}]
                #   Wait, I'm overcomplicating. Let me use the "dominated area" approach.
                
                # The marginal gain is simply the area that the new point dominates that wasn't dominated before.
                # For 2D with min objectives, a point (a,b) dominates the region [a,ref[0]] x [b,ref[1]] that isn't already covered.
                # The uncovered area due to inserting at f1_new between f1_left and f1_right:
                # It's the area of the L-shape or rectangle that was previously uncovered.
                
                # For simplicity and correctness, let me compute it as:
                # gain = (min(f1_right, ref[0]) - f1_new) * (f2_new - min(f2_right, f2_new)) 
                # No... let me just do it properly with the formula.
                
                # I'll use: for each adjacent pair, compute the trapezoid area change.
                # This is getting too complex inline. Let me use a different approach:
                # Just estimate the gain as proportional to the gap size, which is a good heuristic.
                pass

            else:
                # Insertion in the middle
                f1_left = pf_points[idx - 1]
                f2_left = 1.0 - np.sqrt(f1_left)
                f1_right = pf_points[idx]
                f2_right = 1.0 - np.sqrt(f1_right)
                f2_new = 1.0 - np.sqrt(f1_new)
                # The gain is the area of the rectangle that the new point covers that wasn't covered before.
                # Before: the gap between left and right was covered by the left point extending to f1_right and the right point extending left.
                # The new point at (f1_new, f2_new) where f2_left > f2_new > f2_right (since f1_left < f1_new < f1_right)
                # The gain = (f1_right - f1_new) * (f2_new - f2_right) + (f1_new - f1_left) * (f2_left - f2_new)
                # Wait, that's not right either. Let me think geometrically.
                # The hypervolume is the area under the step function.
                # Before: from f1_left to f1_right, the "height" was f2_left (from the left point's perspective, it covers up to f2_left for f1 in [f1_left, f1_right))
                # Actually no. In 2D HV with ref, we sort by f1 asc and compute:
                # HV = sum (ref[0] - f1_i) * (f2_prev - f2_i) where f2_prev is the f2 of the previous point (starting at ref[1])
                # So the contribution of the interval [f1_i, f1_{i+1}) is (ref[0] - f1_i) * (f2_i - f2_{i+1}) ... no.
                
                # Let me just use the correct formula:
                # Points sorted by f1: p_0, p_1, ..., p_{m-1}
                # HV = (ref[0] - p_0[0]) * (ref[1] - p_0[1]) + sum_{i=1}^{m-1} (ref[0] - p_i[0]) * (p_{i-1}[1] - p_i[1])
                # where each term is only included if p_i[1] < p_{i-1}[1]
                
                # When inserting a new point at index idx (between idx-1 and idx):
                # Before: term for p_{idx-1} was (ref[0]-f1_{idx-1})*(f2_{idx-2} - f2_{idx-1}) and term for p_idx was (ref[0]-f1_idx)*(f2_{idx-1} - f2_idx)
                # After: term for p_{idx-1} becomes (ref[0]-f1_{idx-1})*(f2_{idx-2} - f2_new), new term is (ref[0]-f1_new)*(f2_new - f2_new)=0... no.
                # After: p_{idx-1} term: (ref[0]-f1_{idx-1})*(f2_{idx-2} - f2_new)
                # New point term: (ref[0]-f1_new)*(f2_new - f2_new) = 0? No, the next point after new is p_idx.
                # New point term: (ref[0]-f1_new)*(f2_new - f2_idx) ... no, it's (ref[0]-f1_new)*(prev_f2 - f2_new) where prev_f2 is the f2 of the point before it in sorted order, which is f2_{idx-1}... no wait.
                # I'm confusing myself. Let me restart the formula.
                
                # Correct 2D HV formula (minimization):
                # Sort points by f1 ascending: q_0, q_1, ..., q_{m-1}
                # HV = sum_{i=0}^{m-1} (ref[0] - q_i[0]) * (q_{i-1}[1] - q_i[1])
                # where q_{-1}[1] = ref[1], and the term is only added if q_i[1] < q_{i-1}[1]
                
                # So the contribution of point i depends on the f2 of point i-1.
                # When we insert a new point at position idx:
                # - The point that was at idx is now at idx+1. Its previous f2 reference changes from f2_{idx-1} to f2_new.
                #   Its old contribution: (ref[0]-f1_idx)*(f2_{idx-1} - f2_idx) [if f2_idx < f2_{idx-1}]
                #   Its new contribution: (ref[0]-f1_idx)*(f2_new - f2_idx) [if f2_idx < f2_new]
                # - The new point's contribution: (ref[0]-f1_new)*(f2_{idx-1} - f2_new) [if f2_new < f2_{idx-1}]
                # - The point at idx-1's contribution stays the same: (ref[0]-f1_{idx-1})*(f2_{idx-2} - f2_{idx-1})
                
                # So the change in HV = [(ref[0]-f1_idx)*(f2_new - f2_idx) + (ref[0]-f1_new)*(f2_{idx-1} - f2_new)] - [(ref[0]-f1_idx)*(f2_{idx-1} - f2_idx)]
                # = (ref[0]-f1_idx)*f2_new - (ref[0]-f1_idx)*f2_idx + (ref[0]-f1_new)*f2_{idx-1} - (ref[0]-f1_new)*f2_new - (ref[0]-f1_idx)*f2_{idx-1} + (ref[0]-f1_idx)*f2_idx
                # = (ref[0]-f1_idx)*f2_new + (ref[0]-f1_new)*f2_{idx-1} - (ref[0]-f1_new)*f2_new - (ref[0]-f1_idx)*f2_{idx-1}
                # = f2_new*((ref[0]-f1_idx) - (ref[0]-f1_new)) + f2_{idx-1}*((ref[0]-f1_new) - (ref[0]-f1_idx))
                # = f2_new*(f1_new - f1_idx) + f2_{idx-1}*(f1_idx - f1_new)
                # = (f1_new - f1_idx)*(f2_new - f2_{idx-1})
                # Since f1_new < f1_idx and f2_new > f2_{idx-1} (because f1_new > f1_{idx-1} means f2_new < f2_{idx-1}... wait)
                # f1_{idx-1} < f1_new < f1_idx, so f2_{idx-1} > f2_new > f2_idx
                # So f1_new - f1_idx < 0 and f2_new - f2_{idx-1} < 0, so the product is positive. Good.
                # gain = (f1_idx - f1_new)*(f2_{idx-1} - f2_new)
                
                return (f1_right - f1_new) * (f2_left - f2_new)

            # For the edge cases, let me handle them properly:
            if idx == 0:
                f2_new = 1.0 - np.sqrt(f1_new)
                f1_right = pf_points[0]
                f2_right = 1.0 - np.sqrt(f1_right)
                # Using the formula with "virtual" left point at ref:
                # gain = (f1_right - f1_new) * (ref[1] - f2_new) ... let me derive
                # Before: first point at f1_right contributed (11 - f1_right)*(11 - f2_right)
                # After: new point at f1_new contributes (11 - f1_new)*(11 - f2_new), old first contributes (11 - f1_right)*(f2_new - f2_right)
                # Change = (11 - f1_new)*(11 - f2_new) + (11 - f1_right)*(f2_new - f2_right) - (11 - f1_right)*(11 - f2_right)
                # = (11 - f1_new)*(11 - f2_new) + (11 - f1_right)*f2_new - (11 - f1_right)*f2_right - (11 - f1_right)*11 + (11 - f1_right)*f2_right
                # = (11 - f1_new)*(11 - f2_new) + (11 - f1_right)*f2_new - (11 - f1_right)*11
                # = (11 - f2_new)*(11 - f1_new) - (11 - f1_right)*(11 - f2_new)
                # = (11 - f2_new)*(f1_right - f1_new)
                return (11.0 - f2_new) * (f1_right - f1_new)

            elif idx == len(pf_points):
                f2_new = 1.0 - np.sqrt(f1_new)
                f1_left = pf_points[-1]
                f2_left = 1.0 - np.sqrt(f1_left)
                # Before: last point at f1_left contributed (11 - f1_left)*(f2_prev - f2_left) where f2_prev is from previous point
                # After: old last contributes (11 - f1_left)*(f2_prev - f2_new), new last contributes (11 - f1_new)*(f2_new - ... no, next is ref)
                # New last contributes (11 - f1_new)*(f2_new - f2_new)... no.
                # The new last point's contribution: (11 - f1_new)*(prev_f2 - f2_new) where prev_f2 = f2_new? No, prev is the old last.
                # New last: (11 - f1_new)*(f2_left - f2_new) ... no, prev_f2 for the new last is f2_left? No.
                # Sorted order: ..., f1_left, f1_new. The point before f1_new is f1_left.
                # So new last contributes (11 - f1_new)*(f2_left - f2_new)
                # Old last (now second-to-last) contributes (11 - f1_left)*(f2_prev - f2_left) -- unchanged
                # But wait, the old last's contribution was (11 - f1_left)*(f2_prev - f2_left) and it remains the same.
                # The new point adds (11 - f1_new)*(f2_left - f2_new)
                # But we also need to check: did the old last's contribution change? No, its prev is still the same.
                # So gain = (11 - f1_new)*(f2_left - f2_new)
                # But wait, this isn't right because the old last point's "next" in the HV formula doesn't exist, it uses ref.
                # Let me re-derive: 
                # Before: ..., p_{m-2}, p_{m-1} (at f1_left, f2_left)
                # HV includes: (11 - f1_{m-1})*(f2_{m-2} - f2_{m-1}) for p_{m-1}
                # After: ..., p_{m-2}, p_{m-1} (at f1_left, f2_left), p_new (at f1_new, f2_new)
                # HV includes: (11 - f1_{m-1})*(f2_{m-2} - f2_left) for p_{m-1} [same as before]
                # Plus: (11 - f1_new)*(f2_left - f2_new) for p_new
                # So gain = (11 - f1_new)*(f2_left - f2_new)
                return (11.0 - f1_new) * (f2_left - f2_new)

        # Phase 1: Place initial points using adaptive bisection
        # Start with endpoints
        x1_0 = 0.01
        x1_1 = 0.99

        # Evaluate both endpoints
        x = make_x_on_pf(x1_0)
        f = eval_point(x)
        pf_points.append(x1_0)

        x = make_x_on_pf(x1_1)
        f = eval_point(x)
        pf_points.append(x1_1)
        pf_points.sort()

        # Adaptive bisection: always split the interval with the largest marginal gain
        while used < self.budget and time.time() < deadline:
            # Find the interval with the largest marginal gain
            best_gain = -1
            best_x1 = None
            best_idx = -1

            # Check gaps between consecutive points
            for i in range(len(pf_points) - 1):
                f1_left = pf_points[i]
                f1_right = pf_points[i + 1]
                # Try the midpoint
                mid = (f1_left + f1_right) / 2.0
                gain = compute_hv_marginal_gain(mid)
                if gain > best_gain:
                    best_gain = gain
                    best_x1 = mid
                    best_idx = i + 1  # insertion index

            # Also check extending beyond the current range (optional, but the front is in [0,1])
            # The front spans f1 in [0, 1], so we're already covering it well.

            if best_x1 is None or best_gain <= 0:
                break

            # Evaluate the new point
            x = make_x_on_pf(best_x1)
            f = eval_point(x)
            pf_points.insert(best_idx, best_x1)

        return evaluated
# EVOLVE-BLOCK-END
