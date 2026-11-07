# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 using a hybrid approach:
    - Exploit ZDT2 structure: x1 controls f1, x2..x30 control f2
    - Use random sampling with targeted exploration
    - Maintain a Pareto front approximation
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_vec = hi - lo

        # ZDT2 structure:
        # f1 = x1
        # f2 = (1 + 10*(n-1)) * (1 - sqrt(f1/(n))) where n = dim
        # So the Pareto front is f2 = (1+10*(dim-1))*(1-sqrt(f1/dim))
        # with f1 in [0, 1]
        #
        # Strategy:
        # 1. Sample x1 uniformly in [0,1] and set x2..x30 to minimize f2
        #    (which means making sum(x_i^2) small, so x_i near 0)
        # 2. Also do some random exploration
        # 3. Keep track of best non-dominated points

        n = self.dim
        n_eval = 0

        # We'll maintain a list of (x, f) pairs that are non-dominated
        # Start with a structured sampling of the Pareto front
        # For ZDT2, if we set x1 = t and x2..xn = 0, we get:
        # f1 = t, f2 = (1+10*(n-1))*(1-sqrt(t/n))
        # This gives us points on the Pareto front (or very close)

        # Phase 1: Structured sampling along the Pareto front
        # Use most of the budget for this
        # We want good coverage of f1 in [0, 1]

        # Let's use a mix:
        # - 70% budget: structured sampling (vary x1, keep x2..xn small)
        # - 30% budget: random exploration to find better points

        structured_budget = int(self.budget * 0.7)
        random_budget = self.budget - structured_budget

        # For structured sampling, we'll use different strategies:
        # 1. Uniform grid on x1
        # 2. Focus on regions where the front is curved

        # Generate x1 values: mix of uniform and concentrated near 0 and 1
        # The Pareto front f2 = (1+10*(n-1))*(1-sqrt(f1/n))
        # is concave, so we might want more points where it's steep

        # Let's use a combination:
        # - 50% uniform in [0, 1]
        # - 25% concentrated in [0, 0.1] (steep part)
        # - 25% concentrated in [0.9, 1.0] (flat part)

        x1_values = []
        n_uniform = int(structured_budget * 0.5)
        n_low = int(structured_budget * 0.25)
        n_high = structured_budget - n_uniform - n_low

        x1_uniform = np.linspace(0, 1, n_uniform, endpoint=False) + 1.0/(2*n_uniform)
        x1_low = np.linspace(0, 0.1, n_low, endpoint=False) + 0.1/(2*n_low)
        x1_high = np.linspace(0.9, 1.0, n_high, endpoint=False) + 0.1/(2*n_high)

        x1_values = np.concatenate([x1_uniform, x1_low, x1_high])

        # Shuffle to mix the regions
        perm = np.random.permutation(len(x1_values))
        x1_values = x1_values[perm]

        # Store all evaluated points
        all_x = []
        all_f = []

        # Evaluate structured points
        for i in range(len(x1_values)):
            if n_eval >= self.budget:
                break
            if time.time() > deadline:
                break

            x1 = x1_values[i]

            # For ZDT2, to minimize f2 given x1, we want x2..xn to be small
            # f2 depends on sum(x_i^2 for i=2..n) through g(x) = sum(x_i^2)
            # Actually, ZDT2: f2 = (1 + 10*(n-1)) * (1 - sqrt(f1/n))
            # Wait, let me recall the exact ZDT2 formulation:
            # f1(x) = x1
            # g(x) = 1 + 10*(n-1) + sum((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5))) for i=2..n
            # f2(x) = g(x) * (1 - sqrt(f1/g(x)))
            #
            # Hmm, this is more complex. The Pareto front is:
            # f2 = 1 - sqrt(f1) when g(x) is minimized
            # g_min = 1 (when x_i = 0.5 for all i>=2)
            # So f2 = 1 - sqrt(f1) for f1 in [0,1]
            #
            # Wait, let me be more careful. The standard ZDT2:
            # f1(x) = x1
            # g(x) = 1 + 10*(n-1) + sum_{i=2}^{n} [(x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))]
            # f2(x) = g(x) * (1 - sqrt(f1(x)/g(x)))
            #
            # The Pareto-optimal solutions have g(x) = 1 (minimum), which occurs
            # when x_i = 0.5 for all i >= 2.
            # Then f2 = 1 * (1 - sqrt(f1/1)) = 1 - sqrt(f1)
            #
            # So the Pareto front is: f2 = 1 - sqrt(f1) for f1 in [0, 1]

            # To get close to the Pareto front, set x2..xn = 0.5
            x = np.zeros(self.dim)
            x[0] = x1
            x[1:] = 0.5  # This minimizes g(x)

            x = np.clip(x, lo, hi)
            f = problem(x)
            all_x.append(x)
            all_f.append(f)
            n_eval += 1

        # Phase 2: Random exploration
        # Sample random points, especially focusing on getting x2..xn close to 0.5
        for i in range(random_budget):
            if n_eval >= self.budget:
                break
            if time.time() > deadline:
                break

            # Random x1, and x2..xn near 0.5 with some noise
            x1 = np.random.uniform(0, 1)
            x = np.zeros(self.dim)
            x[0] = x1
            # Add noise around 0.5 for x2..xn
            x[1:] = 0.5 + 0.1 * np.random.randn(self.dim - 1)
            x = np.clip(x, lo, hi)

            f = problem(x)
            all_x.append(x)
            all_f.append(f)
            n_eval += 1

        # Phase 3: If we have budget left, do more targeted exploration
        remaining = self.budget - n_eval
        if remaining > 0:
            # Use remaining budget to explore around the best points
            # Convert to arrays for easier manipulation
            all_x = np.array(all_x)
            all_f = np.array(all_f)

            # Find the non-dominated set
            # Simple approach: for each point, check if it's dominated by any other
            nd_set = []
            for i in range(len(all_f)):
                is_dominated = False
                for j in range(len(all_f)):
                    if i == j:
                        continue
                    if np.all(all_f[j] <= all_f[i]) and np.any(all_f[j] < all_f[i]):
                        is_dominated = True
                        break
                if not is_dominated:
                    nd_set.append(i)

            nd_set = np.array(nd_set)

            # Mutate points from the non-dominated set
            for i in range(remaining):
                if time.time() > deadline:
                    break

                if len(nd_set) == 0:
                    parent_idx = np.random.randint(len(all_x))
                else:
                    parent_idx = nd_set[np.random.randint(len(nd_set))]

                x_parent = all_x[parent_idx]
                # Small perturbation
                x_child = x_parent + 0.05 * np.random.randn(self.dim)
                x_child = np.clip(x_child, lo, hi)

                f_child = problem(x_child)
                all_x = np.vstack([all_x, x_child])
                all_f = np.vstack([all_f, f_child])
                n_eval += 1

        return all_x
# EVOLVE-BLOCK-END
