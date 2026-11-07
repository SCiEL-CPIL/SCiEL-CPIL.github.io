import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0
        
        n = self.dim
        budget = self.budget
        
        # DTLZ7 structure:
        # f1 = x[0]
        # f2 = x[1]
        # f3 = (1 + g(x[2:])) * (1 + sum_{i=1}^{n-2} (1+x_i) * (1 - x_i/(1+x_i)))
        # 
        # The Pareto front is located at x[2:] = 0 (so g=0).
        # On the Pareto front:
        # f1 = x[0]
        # f2 = x[1]
        # f3 = 1 + sum_{i=1}^{n-2} (1+x_i) * (1 - x_i/(1+x_i))
        #     = 1 + sum_{i=1}^{n-2} (1 - x_i^2/(1+x_i))
        #
        # Reference point: [1.1, 1.1, 25.0]
        #
        # We want to maximize hypervolume.
        # The "corner" points that are likely useful:
        # 1. x = [0, 0, 0, ..., 0] -> f = [0, 0, 1]
        #    Volume contribution: (1.1-0)*(1.1-0)*(25-1) = 1.21 * 24 = 29.04
        #
        # 2. x = [1, 1, 0, ..., 0] -> f = [1, 1, 1]
        #    This is dominated by point 1? No, f1=1 < 1.1, f2=1 < 1.1, f3=1 < 25.
        #    But point 1 dominates point 2? f1=0<1, f2=0<1, f3=1=1. Yes, point 1 dominates point 2.
        #    So point 2 adds no volume.
        #
        # Wait, let's re-read the scoring.
        # "Only points strictly better than the reference point in every objective count."
        # Hypervolume is calculated over the non-dominated set of recorded points.
        #
        # If we only record [0,0,1], the HV is (1.1)*(1.1)*(24) = 29.04.
        #
        # Can we get more?
        # We need points that are not dominated by [0,0,1].
        # To not be dominated, a point must have at least one objective strictly smaller than [0,0,1].
        # f1 >= 0, f2 >= 0. So f1 or f2 cannot be smaller than 0.
        # Therefore, we must have f3 < 1.
        #
        # f3 = (1+g) * h.
        # Min f3 is 1 (when x[2:]=0 and x[1..n-2]=0? No.)
        # Let's check f3 again.
        # DTLZ7 f3 = (1 + g(x_M)) * h(y, g)
        # h(y, g) = 1 + sum_{i=1}^{M-1} (1+y_i) * (1 - y_i/(1+y_i))
        # If y_i = 0, term is 1*1 = 1. Sum is M-1. h = 1 + M-1 = M.
        # Wait, standard DTLZ7 definition:
        # f_i(x) = x_{i-1} for i = 1, ..., m-1
        # f_m(x) = (1 + g(x_M)) h(y, g(x_M))
        # where y = (x_{m-1}, ..., x_{m+M-2})
        # h(y, g) = 1 + sum_{i=1}^{M-1} (1+y_i) (1 - y_i / (1+y_i))
        #
        # For m=3, M=20.
        # f1 = x0, f2 = x1.
        # f3 = (1+g) * (1 + sum_{j=1}^{19} (1+x_{1+j}) (1 - x_{1+j}/(1+x_{1+j})))
        #
        # If x_{2..21} = 0, g=0.
        # Sum term: j=1..19. x_{1+j} = x_2..x_20.
        # If x_2..x_20 = 0, term is 1. Sum = 19.
        # h = 1 + 19 = 20.
        # f3 = 1 * 20 = 20.
        #
        # So x=[0,0,0...0] gives f=[0,0,20].
        # HV = 1.1 * 1.1 * (25-20) = 1.21 * 5 = 6.05.
        #
        # The current program scores 1.68. The reference HV (dense sample) must be small.
        #
        # How to minimize f3?
        # f3 = (1+g) * h.
        # To minimize f3, we need small g and small h.
        # g(x_M) = sum (x_i - 0.35 sin(10 pi x_i))^2 for i in M.
        # Min g = 0 at x_i = 0.35 sin(10 pi x_i) or x_i=0 (if we are careful). Actually x=0 is a local min for g?
        # g(0) = 0.
        #
        # To minimize h, we need to minimize sum (1+y_i)(1 - y_i/(1+y_i)).
        # Let t(y) = (1+y)(1 - y/(1+y)) = (1+y) - y = 1.
        # Wait. (1+y) * ( (1+y-y)/(1+y) ) = (1+y) * (1/(1+y)) = 1.
        # So each term is exactly 1?
        # Then h = 1 + sum(1) = 1 + (M-1) = M.
        # This implies h is constant M for all y in [0,1]?
        # Let's check the derivative or values.
        # y=0 -> 1*1 = 1.
        # y=1 -> 2*(1-1/2) = 2*0.5 = 1.
        # Yes, h is constant M = 20.
        #
        # So f3 = (1+g(x[2:])) * 20.
        # To minimize f3, we must minimize g(x[2:]).
        # Min g is 0.
        # So min f3 = 20.
        #
        # This means the Pareto front is a flat line/curve at f3=20?
        # No, x0 and x1 vary independently.
        # f1 = x0, f2 = x1.
        # f3 = 20 (if x[2:] is on the manifold where g=0).
        #
        # So the Pareto front is the line segment from (0,0,20) to (1,1,20).
        #
        # Reference point: (1.1, 1.1, 25).
        #
        # Points on the front: (x0, x1, 20) for x0,x1 in [0,1].
        #
        # Hypervolume of the front:
        # The dominated region is the union of boxes [0, x0] x [0, x1] x [20, 25] for all x0,x1 in [0,1].
        # This is simply the box [0,1] x [0,1] x [20,25].
        # Volume = 1 * 1 * 5 = 5.
        #
        # Wait, is that right?
        # Any point (f1, f2, 20) dominates the box [0, f1] x [0, f2] x [20, 25].
        # The union of all such boxes as f1, f2 vary in [0,1] is [0,1] x [0,1] x [20,25].
        # Volume = 5.
        #
        # If we just pick x=[0,0,0...0], we get f=[0,0,20].
        # Box: [0,0] x [0,0] x [20,25] -> Volume 0.
        #
        # We need to cover the area in f1-f2 plane.
        # To maximize HV, we want to cover the square [0,1]x[0,1] as much as possible.
        # Each point (f1, f2, 20) covers rectangle [0, f1]x[0, f2].
        #
        # We have budget 200.
        # We should sample the f1-f2 plane.
        # Since f3 is constant 20 (if we set x[2:] correctly), the HV is 5 * Area_Covered_in_f1f2.
        #
        # How to set x[2:] to make g=0?
        # g(x) = sum (x_i - 0.35 sin(10 pi x_i))^2.
        # We need x_i such that x_i = 0.35 sin(10 pi x_i).
        # x=0 is a solution.
        # Are there others?
        # 0.35 sin(10 pi x). Max is 0.35.
        # For x in [0, 0.35], sin(10 pi x) oscillates.
        # We can just use x=0.
        #
        # So strategy:
        # 1. Set x[2:] = 0.
        # 2. Vary x[0] and x[1] to cover [0,1]x[0,1].
        #
        # We want to maximize the area of the union of rectangles [0, x0] x [0, x1].
        # This is equivalent to placing points to maximize the "staircase" area.
        #
        # A grid of points will work.
        # With 200 points, a 14x14 grid (196 points) is good.
        # Or we can be smarter.
        #
        # The reference HV is 5.
        # If we cover the whole square, HV = 5.
        # Score = HV / Ref_HV.
        # If Ref_HV is calculated from a "dense fixed sample", it should be close to 5.
        #
        # Let's verify the current score 1.68.
        # If current program only evaluates x=0, it gets f=[0,0,20].
        # HV = 0.
        # Score 0.
        # But it scores 1.68.
        # Why?
        # "Every returned objective vector is recorded".
        # The loop evaluates x_zero 200 times.
        # All are [0,0,20].
        # Non-dominated set: { [0,0,20] }.
        # HV = 0.
        #
        # Maybe my understanding of DTLZ7 is slightly off or the reference point/HV calc is different.
        # Or maybe x=0 doesn't give f3=20?
        #
        # Let's re-read the task description carefully.
        # "minimizing all 3 objectives of DTLZ7"
        #
        # If the current program scores 1.68, it must be getting some HV.
        # If it only has [0,0,20], HV is 0.
        # Unless... the problem implementation is different?
        # "problem(x) clips ... and returns 3 objective values."
        #
        # Maybe the "dense fixed sample" has a very small HV?
        # If the reference sample is sparse or bad, Ref_HV could be small.
        # But "dense fixed sample of the known Pareto front".
        #
        # Let's assume the standard DTLZ7.
        # If I sample a grid in x0, x1, I should get HV close to 5.
        # If Ref_HV is 5, score ~1.
        # If current score is 1.68, maybe Ref_HV is smaller?
        # Or maybe current program is doing something else?
        # No, it evaluates x_zero repeatedly.
        #
        # Is it possible that f3 is NOT 20 at x=0?
        # f3 = (1+g) * h.
        # If h is not constant?
        # h(y) = 1 + sum (1+y_i)(1 - y_i/(1+y_i)).
        # I proved this is 1 for each term.
        # So h = M = 20.
        #
        # Wait, is M=20?
        # dim=22, n_obj=3.
        # M = dim - n_obj + 1 = 22 - 3 + 1 = 20.
        # Yes.
        #
        # So f3=20 at x=0.
        #
        # Why score 1.68?
        # Maybe the reference point is different? [1.1, 1.1, 25.0].
        #
        # Let's try the grid strategy. It should definitely be better than a single point.
        
        # Strategy:
        # 1. Set x[2:] = 0 to ensure g=0 and f3=20.
        # 2. Sample x[0], x[1] on a grid to cover [0,1]x[0,1].
        # 3. Use remaining budget to jitter or refine if needed, but grid should be sufficient.
        
        # Number of points in grid
        # Budget 200.
        # Let's use a 14x14 grid = 196 points.
        # Plus 4 corners explicitly? They are in the grid if we include 0 and 1.
        
        n_grid = 14
        # Ensure we don't exceed budget
        n_points = n_grid * n_grid
        if n_points > budget:
             n_grid = int(np.sqrt(budget))
             n_points = n_grid * n_grid
             
        xs0 = np.linspace(0, 1, n_grid)
        xs1 = np.linspace(0, 1, n_grid)
        
        best_x = np.zeros(n)
        
        # Precompute x[2:] = 0
        x_base = np.zeros(n)
        
        count = 0
        for x0 in xs0:
            for x1 in xs1:
                if time.time() > deadline - 10.0:
                    break
                if count >= budget:
                    break
                
                x = x_base.copy()
                x[0] = x0
                x[1] = x1
                
                # Evaluate
                # We don't store F explicitly as the evaluator records it
                problem(x)
                
                # Update best? 
                # "return value of __call__ is ignored"
                # But we must return something.
                best_x = x
                
                count += 1
            if time.time() > deadline - 10.0:
                break
            if count >= budget:
                break
        
        # Use remaining budget if any
        remaining = budget - count
        for i in range(remaining):
             if time.time() > deadline - 5.0:
                 break
             # Jitter around best or random?
             # Random might find better spots if my model is wrong.
             # But if model is right, grid is optimal.
             # Let's just evaluate x_base (0,0,0...) again? No, already covered.
             # Let's try random points in x0, x1 to be safe.
             x_rand = x_base.copy()
             x_rand[0] = np.random.rand()
             x_rand[1] = np.random.rand()
             problem(x_rand)
        
        return best_x
