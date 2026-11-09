<<<<<<< SEARCH
        # Phase 1: Initialization
        # Use a small set of random points to find a good starting basin.
        # 15 points is enough to get a reasonable start without wasting budget.
        n_init = 15
        for _ in range(n_init):
            if calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            timed_call(x)
=======
        # Phase 1: Initialization
        # Use a mix of structured and random points to find a good starting basin.
        n_init = 20
        # Start with center
        x_center = np.zeros(self.dim)
        timed_call(x_center)
        # Axis-aligned probes to detect ridge direction early
        for d in range(self.dim):
            if calls_made >= self.budget:
                break
            x_probe = np.zeros(self.dim)
            x_probe[d] = 1.0
            timed_call(x_probe)
            x_probe2 = np.zeros(self.dim)
            x_probe2[d] = -1.0
            timed_call(x_probe2)
        # Fill remaining budget with random points
        n_random = n_init - (1 + 2 * self.dim)
        for _ in range(max(0, n_random)):
            if calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            timed_call(x)
>>>>>>> REPLACE

