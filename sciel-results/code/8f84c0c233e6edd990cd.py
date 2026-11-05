import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n = 256
    start_time = time.time()
    deadline = 45.0  # Safety margin from 60s limit
    
    # 1. Generate all 3-term arithmetic progressions (a, b, c) with a < b < c
    # b = (a+c)/2 => c = 2b - a
    # We need x_a + x_b + x_c <= 2
    # Variables x_0 ... x_{255} correspond to values 1 ... 256
    
    # Greedy initial solution
    # Sort candidates by reciprocal value descending (i.e., index ascending)
    # We want to include small numbers if possible.
    # However, small numbers participate in many APs.
    # Let's try a simple greedy: iterate 1 to n, add if no conflict with existing.
    # This is what the seed does. Let's try a slightly different greedy:
    # Iterate 1 to n. If adding x creates an AP with two existing elements, skip.
    # This is equivalent to the seed.
    
    # Let's try to get a better initial solution by prioritizing smaller numbers
    # but checking conflicts.
    
    selected = []
    # Precompute for each i, the pairs (j, k) such that i is part of an AP with j, k?
    # Or just check on the fly.
    # For speed in greedy, we can maintain a set of selected indices.
    # Check if adding x creates an AP:
    # 1. x is the middle: x = (a+c)/2 => a, c in selected and a+c=2x
    # 2. x is the end: x = a+2d => a, a+d in selected and a+d+x=2(a+d)? No.
    #    If x is the largest: x = a+2d. Then a, a+d must be in selected.
    #    So for each d, if x-2d in selected and x-d in selected, conflict.
    # 3. x is the smallest: x = c-2d. Then c, c-d in selected.
    #    So for each d, if x+2d in selected and x+d in selected, conflict.
    
    # To make greedy faster, we can just use the seed's logic which is O(N^3) worst case but N=256 is small.
    # Seed:
    # for x in range(1, n+1):
    #    if all(2*b-a != x for a in selected for b in selected if a < b):
    #        selected.append(x)
    # This checks if x is the middle of an AP with two existing elements.
    # Wait, the seed logic `2*b-a != x` checks if x is the *third* term?
    # Let's trace: a, b in selected, a < b.
    # If 2*b - a == x, then a, b, x form an AP?
    # a, b, x => b-a = x-b => 2b = a+x => x = 2b-a. Yes.
    # So the seed checks if x is the end of an AP starting with two existing elements.
    # Does it check if x is the middle?
    # If a, x, b is an AP, then x = (a+b)/2.
    # The seed does NOT check this.
    # Example: selected = [1, 3]. Next x=2.
    # a=1, b=3. 2*3 - 1 = 5 != 2. So 2 is added.
    # But 1, 2, 3 is an AP!
    # So the seed program is INVALID?
    # Let's re-read the seed code carefully.
    # `if all(2*b-a!=x for a in selected for b in selected if a<b)`
    # If selected = [1, 3], a=1, b=3. 2*3-1 = 5. 5!=2. Condition holds.
    # selected becomes [1, 3, 2].
    # This contains 1, 2, 3 which is an AP.
    # The task requires "without any nonconstant k=3 term arithmetic progression".
    # So the seed program is likely producing invalid sets or the evaluator is lenient?
    # The prompt says "score is its floating serialization... invalid result... scores 0".
    # But the seed score is 1.0000.
    # Maybe I am misinterpreting the seed code or the AP definition.
    # "nonconstant k=3 term arithmetic progression" usually means a, a+d, a+2d with d!=0.
    # If the seed produces invalid sets, why is it 1.0?
    # Perhaps the evaluator checks validity and if invalid, score is 0?
    # But it says "CURRENT PROGRAM (score 1.0000)".
    # Let's check the seed code again.
    # Maybe `a<b` and `2*b-a` covers all cases?
    # If x is the middle, say x=2, selected=[1,3].
    # Is there a pair (a,b) in selected such that 2*b-a == x?
    # Pairs: (1,3). 2*3-1 = 5. No.
    # So 2 is added.
    # Set: {1, 2, 3}.
    # This is an AP.
    # So the seed program IS flawed if the evaluator enforces validity strictly.
    # However, the prompt says "score 1.0000".
    # It is possible the evaluator only checks the objective value and assumes validity, or the "seed" provided in the prompt is just a placeholder and not actually evaluated?
    # "Keep the function names and signatures of the seed program."
    # "The evaluator measures the raw value... invalid result... scores 0".
    # If the seed scores 1.0, it must be valid.
    # Let's re-read the condition: `all(2*b-a!=x for a in selected for b in selected if a<b)`
    # This condition ensures that x is not the *third* element of an AP where the first two are in `selected`.
    # It does NOT prevent x from being the *middle* element.
    # It does NOT prevent x from being the *first* element.
    # If the set is built incrementally, and we only check if the new element completes an AP with two *existing* elements as the *last* element, we miss cases where the new element is the middle or first.
    #
    # Wait, if we add elements in increasing order 1, 2, ..., n:
    # If we add x, and a, b are already in selected, then a < x and b < x.
    # The only AP involving x and two smaller numbers is where x is the largest.
    # Because if x is the middle, one number is smaller and one is larger. But the larger one hasn't been added yet.
    # If x is the smallest, both others are larger, not added yet.
    # So, if we iterate x from 1 to n, and only check if x forms an AP with two *previous* elements (which must be smaller than x), then the only possible AP is (a, b, x) with a < b < x.
    # In this case, x is the third term.
    # Condition: a, b, x in AP => x - b = b - a => x = 2b - a.
    # So the seed code IS correct for generating a valid set, provided we iterate in increasing order and only check against previously selected elements.
    # My counter-example [1, 3, 2] failed because 2 was added *after* 3?
    # No, the seed iterates `for x in range(1, 257)`.
    # So x=1, selected=[1].
    # x=2. selected=[1]. Pairs: none. Add 2. selected=[1, 2].
    # x=3. selected=[1, 2]. Pairs: (1,2). 2*2-1 = 3. 3==3. Condition fails. Do NOT add 3.
    # So 3 is NOT added.
    # So the seed does NOT produce [1, 2, 3]. It produces [1, 2, 4, ...].
    # Okay, my manual trace was wrong because I assumed the order of selection allowed 3 before 2. But the loop is sequential.
    # So the seed program is valid.
    
    # Now, back to the BIP approach.
    # We will generate all AP constraints.
    # Constraint: x_a + x_b + x_c <= 2 for all a < b < c in AP.
    
    # Build constraint matrix A_ub and b_ub
    # Number of variables: 256
    # Number of constraints: ~16k
    
    ap_constraints = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            c = 2 * b - a
            if c > n:
                break
            if c > b:
                # a, b, c form an AP
                # Indices are 0-based in the vector
                # x_a is at index a-1
                ap_constraints.append((a - 1, b - 1, c - 1))
    
    # Create A_ub matrix
    # Since the number of constraints is small, we can build a sparse matrix or dense.
    # Dense 16k x 256 is ~4 million floats, ~32 MB. Manageable.
    # But scipy.optimize.milp prefers sparse or structured.
    # Let's use sparse COO format.
    
    import scipy.sparse as sp
    
    rows = []
    cols = []
    vals = []
    
    for idx, (ia, ib, ic) in enumerate(ap_constraints):
        rows.extend([idx, idx, idx])
        cols.extend([ia, ib, ic])
        vals.extend([1, 1, 1])
        
    A_ub = sp.coo_matrix((vals, (rows, cols)), shape=(len(ap_constraints), n)).tocsr()
    b_ub = np.ones(len(ap_constraints))
    
    # Objective: Maximize sum(1/i * x_i)
    # Minimize -sum(1/i * x_i)
    c = -np.array([1.0 / i for i in range(1, n + 1)])
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all integer
    integrality = np.ones(n)
    
    # Initial solution from greedy (seed logic)
    # This provides a feasible starting point, which can help the solver.
    # However, scipy.optimize.milp doesn't take an initial solution directly in the standard API easily?
    # It does not have a `x0` parameter.
    # So we just solve from scratch.
    
    # Run MILP
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A_ub, ub=b_ub),
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": deadline - (time.time() - start_time), "mip_rel_gap": 1e-4}
        )
        
        if res.success and res.x is not None:
            # res.x is the solution vector
            # Check if it's close to integer
            x_sol = np.round(res.x)
            # Verify validity just in case
            # If valid, use it
            selected = [int(i + 1) for i in range(n) if x_sol[i] > 0.5]
            
            # Double check validity
            is_valid = True
            sel_set = set(selected)
            for a in selected:
                for b in selected:
                    if a >= b: continue
                    c_val = 2 * b - a
                    if c_val in sel_set:
                        is_valid = False
                        break
                if not is_valid: break
            
            if is_valid:
                return selected
            else:
                # Fallback to greedy
                pass
        else:
            # Fallback to greedy
            pass
            
    except Exception as e:
        # Fallback to greedy
        pass

    # Greedy Fallback
    selected = []
    for x in range(1, n + 1):
        valid = True
        # Check if x is the third term of an AP with two existing elements
        # Since x is the largest, we only need to check pairs (a,b) in selected
        # such that 2*b - a == x
        # Optimization: for each b in selected, a = 2*b - x. Check if a in selected and a < b.
        for b in selected:
            a = 2 * b - x
            if a > 0 and a < b and a in selected:
                valid = False
                break
        if valid:
            selected.append(x)
            
    return selected
# EVOLVE-BLOCK-END
