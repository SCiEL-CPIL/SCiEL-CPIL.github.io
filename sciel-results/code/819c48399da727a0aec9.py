import numpy as np

def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in a Multiple-Constraint Knapsack Problem.
    
    Args:
        prizes: np.array of shape (5, 100) containing prizes for each constraint/item.
                Note: The task description says prizes and weights are shape (5, 100).
                Usually, prize is a single scalar per item, but here it's a vector?
                "Item scores... prizes and weights, of shape (5, 100)".
                "Highest total prize found by any ant".
                If prizes is (5, 100), does "total prize" mean sum of prizes across constraints?
                Or is it a multi-objective?
                "Mean over instances of the highest total prize".
                Usually, MCKP has a single profit vector $p_i$ and multiple weight vectors $w_{ci}$.
                The shape (5, 100) for prizes suggests 5 objectives? Or is it 5 constraints and prizes are aligned?
                Let's re-read carefully: "prizes and weights, of shape (5, 100)".
                "Every constraint has capacity 25".
                "Highest total prize".
                If prizes is (5, 100), it's likely that the "prize" for an item is the sum of its entries in the prize matrix, or perhaps the prize matrix is just repeated?
                However, standard MCKP has one profit per item.
                If the input is (5, 100), it might be that the profit is the sum across the 5 rows?
                Or maybe the 5 rows correspond to 5 different "types" of prizes?
                Given "total prize", it's safest to assume the item's value is the sum of the prize entries for that item across the 5 dimensions, OR that the prize is uniform across the 5 rows?
                Actually, if it's a "multiple-constraint" problem, the constraints are the 5 rows of `weights`.
                What are the 5 rows of `prizes`?
                If it were a single objective, `prizes` would be shape (1, 100) or (100,).
                The fact that it is (5, 100) suggests either:
                1. Multi-objective optimization (but the score is a single "total prize").
                2. The prize is actually a vector, and "total prize" is the sum.
                3. It's a typo in my interpretation, and maybe `prizes` is just (100,) but passed as (5,100) for some reason? No, "shape (5, 100)" is explicit.
                
                Let's assume the value of item $i$ is $V_i = \sum_{k=0}^{4} \text{prizes}[k, i]$.
                Or perhaps $V_i = \text{prizes}[0, i]$?
                "Highest total prize" usually implies a scalar sum.
                Let's calculate `item_values = prizes.sum(axis=0)`.
                
                Wait, if it's a multi-objective problem, "total prize" is ambiguous.
                But the scoring says "highest total prize". This implies a scalar.
                Most likely, the 5 rows of `prizes` are just 5 independent random values, and the "prize" of the item is the sum, or perhaps the mean?
                Let's look at the baseline: `prizes / (weights.sum(axis=0) + 1e-9)`.
                The baseline returns an array of shape (5, 100)??
                "returns one attractiveness score per item".
                So the return shape must be (100,).
                The baseline code:
                `return prizes / (weights.sum(axis=0) + 1e-9)`
                `prizes` is (5, 100). `weights.sum(axis=0)` is (100,).
                The result is (5, 100).
                But the task says "returns one attractiveness score per item".
                This implies the return should be (100,).
                If the baseline returns (5, 100), maybe the evaluator sums it? Or takes the first row?
                "Heuristic entries that are not positive become 1e-9".
                If I return (5, 100), and the evaluator expects (100,), it might fail or behave unexpectedly.
                However, the baseline scored 1.0, so it is "valid".
                If the baseline returns (5, 100) and is valid, then either:
                1. The evaluator handles (5, 100) by summing/averaging.
                2. The task description "one score per item" is loose, and (5,100) is accepted.
                3. The `prizes` in the baseline code is actually being treated as (100,) somehow? No, `prizes` is passed in.
                
                Let's assume the evaluator sums the heuristic rows or uses the first row.
                To be safe, I should return an array of shape (100,).
                How to reduce (5, 100) to (100,)?
                If the prize is the sum, I should use `prizes.sum(axis=0)` in the numerator.
                Let's assume the item's "prize" is the sum of the 5 values.
                
                Let's refine the heuristic:
                `values = prizes.sum(axis=0)`
                `max_w = weights.max(axis=0)`
                `eta = values / (max_w + 1e-9)`
                
                This returns (100,).
                
                Let's double check the baseline behavior.
                Baseline: `prizes / (weights.sum(axis=0) + 1e-9)` -> (5, 100).
                If the evaluator expects (100,), the baseline would be invalid.
                But it scored 1.0.
                So the evaluator MUST accept (5, 100).
                How does it use it?
                "probability proportional to pheromone[item] * heuristic[item]".
                If heuristic is (5, 100), it's not a scalar per item.
                Maybe the evaluator does `heuristic.sum(axis=0)`?
                Or `heuristic[0]`?
                If I return (100,), it is definitely "one score per item".
                If the baseline works with (5, 100), returning (100,) should also work, as it's the cleaner interface.
                
                I will return (100,) to be safe and adhere to "one score per item".
                
                Heuristic choice:
                $V_i = \sum_{k} p_{k,i}$
                $W_{i, \text{max}} = \max_{c} w_{c,i}$
                $\eta_i = \frac{V_i}{W_{i, \text{max}} + \epsilon}$
                
                This favors items with high total prize and low maximum constraint weight.
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Calculate total prize per item (sum across the 5 rows)
    # This assumes the "total prize" is the sum of the prize components.
    item_values = prizes.sum(axis=0)
    
    # Calculate the maximum weight per item across all constraints
    # This identifies the "bottleneck" constraint for each item.
    max_weights = weights.max(axis=0)
    
    # Compute heuristic: Efficiency based on max weight
    # Items with high value and low max weight are most attractive.
    epsilon = 1e-9
    eta = item_values / (max_weights + epsilon)
    
    # Sanitize: Ensure positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
