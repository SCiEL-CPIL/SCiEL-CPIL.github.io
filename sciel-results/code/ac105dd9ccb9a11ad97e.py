"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-trajectory adaptive step-size (1+1)-ES with Halton seeding for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _halton(self, idx, base):
        """Compute the Halton sequence value at index idx for a given base."""
        f = 1.0
        r = 0.0
        i = idx
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def _halton_point(self, idx, dim):
        """Generate a Halton sequence point in [0, 1]^dim."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        return np.array([self._halton(idx + 1, primes[d]) for d in range(dim)])

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        # Phase 1: Halton space-filling to find initial basins
        halton_n = min(50, budget)
        best_x = None
        best_f = np.inf
        candidates = []

        for i in range(halton_n):
            if time.time() > deadline:
                break
            h = self._halton_point(i, dim)
            x = -5.0 + 10.0 * h
            fx = func(x)
            candidates.append((x.copy(), fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Multi-trajectory (1+1)-ES with diverse step sizes
        # Sort candidates and pick top 5 diverse starting points
        candidates.sort(key=lambda t: t[1])
        
        # Select top 5 but ensure some diversity by taking every k-th from sorted
        # Actually, just take top 5 for now, they are likely in the same basin or nearby
        # To ensure diversity, let's take the best, and then 4 random from the next 20
        top1 = candidates[0]
        rest = candidates[1:21]
        if len(rest) >= 4:
            idxs = np.random.choice(len(rest), 4, replace=False)
            extra_starts = [rest[i] for i in idxs]
        else:
            extra_starts = rest

        starts = [top1] + extra_starts
        starts = [s[0].copy() for s in starts]

        # Initialize trajectories
        # Each trajectory has: x, fx, sigma, history of improvements
        trajectories = []
        calls_made = halton_n
        
        for i, sx in enumerate(starts):
            # Assign diverse initial sigmas
            sigmas = [0.5, 1.0, 2.0, 4.0, 8.0]
            sig = sigmas[i % len(sigmas)]
            # If we already evaluated this point in Halton phase, reuse fx
            # But to be safe and simple, we might re-evaluate or just assume it's close
            # Actually, we should check if this exact point was evaluated.
            # Simpler: just evaluate the starting point now if not the best one (which we have fx for)
            if i == 0:
                fx = top1[1]
            else:
                if calls_made >= budget or time.time() > deadline:
                    break
                fx = func(sx)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = sx.copy()
            
            trajectories.append({
                'x': sx.copy(),
                'fx': fx,
                'sigma': sig,
                'last_improve_step': calls_made,
                'best_fx': fx
            })

        # Main loop: allocate budget dynamically
        # We will run until budget is exhausted or deadline
        # Strategy: each step, pick the trajectory with the highest "momentum"
        # Momentum = (best_global - traj_best) / traj_sigma? No, let's use recent improvement.
        
        # Simpler robust strategy:
        # 1. Always keep the global best.
        # 2. Run a round-robin of trajectories, but if a trajectory improves global best,
        #    boost its weight.
        
        # Let's use a weighted selection based on how recently it improved and its current quality.
        # Weight = 1 / (1 + (current_step - last_improve_step)) * (1 / (1 + fx))
        
        while calls_made < budget and time.time() < deadline:
            if not trajectories:
                break
            
            # Calculate weights for selection
            # We want to favor trajectories that are close to the best and have been improving
            # Let's use a simple heuristic:
            # Score = (best_global - traj_fx) + 0.1 * (calls_made - traj_last_improve_step)
            # Actually, we want to explore trajectories that are promising.
            # Let's just do a random selection among the top 2 best trajectories,
            # and occasionally restart the worst one.
            
            # Sort trajectories by current fx
            trajectories.sort(key=lambda t: t['fx'])
            
            # Pick the best trajectory with probability 0.7, second best with 0.3
            if len(trajectories) > 1 and np.random.rand() < 0.3:
                traj = trajectories[1]
            else:
                traj = trajectories[0]
            
            # Generate candidate
            y = np.clip(traj['x'] + traj['sigma'] * np.random.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < traj['fx']:
                # Accept
                traj['x'] = y
                traj['fx'] = fy
                traj['last_improve_step'] = calls_made
                traj['sigma'] *= 1.1  # Increase step size slightly
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            else:
                # Reject
                traj['sigma'] *= 0.9  # Decrease step size
                # If step size gets too small, maybe restart this trajectory?
                if traj['sigma'] < 0.01:
                    # Restart from a random point near the global best?
                    # Or just reset sigma
                    traj['sigma'] = 1.0
                    # Optionally move x to a random point in the box
                    # traj['x'] = np.random.uniform(-5, 5, dim)
                    # traj['fx'] = func(traj['x'])
                    # calls_made += 1
                    # But this is risky if budget is low. Let's just keep x and reset sigma.
            
            # Periodic diversity injection:
            # Every 50 calls, if the worst trajectory is far from the best, restart it
            if calls_made % 50 == 0 and len(trajectories) > 0:
                worst = trajectories[-1]
                if worst['fx'] > best_f + 1.0: # If it's significantly worse
                    # Restart it at a random point
                    if calls_made < budget and time.time() < deadline:
                        new_x = np.random.uniform(-5, 5, dim)
                        new_fx = func(new_x)
                        calls_made += 1
                        worst['x'] = new_x
                        worst['fx'] = new_fx
                        worst['sigma'] = 2.0
                        worst['last_improve_step'] = calls_made
                        if new_fx < best_f:
                            best_f = new_fx
                            best_x = new_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
