import numpy as np
import itertools
import time
import random

N = 17

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin under 90s

    # Pre-generate all 131,072 Boolean inputs once
    inputs = np.array(list(itertools.product([0, 1], repeat=N)), dtype=np.int8)

    # Small subset for fast inner-loop verification
    rng = np.random.default_rng(12345)
    idx = rng.choice(len(inputs), 3000, replace=False)
    subset_inputs = inputs[idx]

    def verify_subset(net):
        if not net:
            return 0
        state = subset_inputs.copy()
        for i, j in net:
            ci = state[:, i]
            cj = state[:, j]
            state[:, i] = np.minimum(ci, cj)
            state[:, j] = np.maximum(ci, cj)
        diffs = np.diff(state, axis=1)
        return int(np.any(diffs < 0, axis=1).sum())

    def verify_full(net):
        if not net:
            return 0
        state = inputs.copy()
        for i, j in net:
            ci = state[:, i]
            cj = state[:, j]
            state[:, i] = np.minimum(ci, cj)
            state[:, j] = np.maximum(ci, cj)
        diffs = np.diff(state, axis=1)
        return int(np.any(diffs < 0, axis=1).sum())

    # Initial: odd-even transposition (136 comparators)
    def generate_oe():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]

    curr_net = generate_oe()
    curr_sub_fail = verify_subset(curr_net)
    curr_cost = len(curr_net) + 100 * curr_sub_fail

    best_net = curr_net.copy()
    best_cost = curr_cost
    best_full_fail = verify_full(curr_net)

    def cost_of(net, sub_fail):
        return len(net) + 100 * sub_fail

    def try_improve(net, sub_fail, cost):
        """If sub_fail==0, do full verify; if valid and better, update best."""
        nonlocal best_net, best_cost, best_full_fail
        if sub_fail == 0:
            ff = verify_full(net)
            if ff == 0:
                c = len(net)
                if c < best_cost:
                    best_cost = c
                    best_net = net.copy()
                    best_full_fail = 0
        return cost

    # Neighbor generators
    def neighbor_add(net):
        new = net.copy()
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        # Insert at a random position (depth-aware)
        pos = random.randint(0, len(new))
        new.insert(pos, (i, j))
        return new

    def neighbor_add_multi(net, k):
        new = net.copy()
        for _ in range(k):
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(new))
            new.insert(pos, (i, j))
        return new

    def neighbor_remove(net):
        if not net:
            return net
        new = net.copy()
        idx = random.randint(0, len(new) - 1)
        new.pop(idx)
        return new

    def neighbor_remove_multi(net, k):
        new = net.copy()
        for _ in range(k):
            if new:
                idx = random.randint(0, len(new) - 1)
                new.pop(idx)
        return new

    def neighbor_move(net):
        if len(net) < 2:
            return net
        new = net.copy()
        a = random.randint(0, len(new) - 1)
        b = random.randint(0, len(new) - 1)
        new[a], new[b] = new[b], new[a]
        return new

    def neighbor_retarget(net):
        if not net:
            return net
        new = net.copy()
        idx = random.randint(0, len(new) - 1)
        i, j = new[idx]
        new_i = random.randint(0, N - 2)
        new_j = random.randint(new_i + 1, N - 1)
        new[idx] = (new_i, new_j)
        return new

    def neighbor_retarget_pair(net):
        """Change both endpoints of a comparator to a new random pair."""
        if not net:
            return net
        new = net.copy()
        idx = random.randint(0, len(new) - 1)
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        new[idx] = (i, j)
        return new

    def neighbor_insert_at_depth(net):
        """Add a comparator but bias position toward early (low depth)."""
        new = net.copy()
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        # Bias toward front
        pos = random.randint(0, max(1, len(new) // 3))
        new.insert(pos, (i, j))
        return new

    def neighbor_swap_adjacent(net):
        """Swap two adjacent comparators (local reorder)."""
        if len(net) < 2:
            return net
        new = net.copy()
        a = random.randint(0, len(new) - 2)
        new[a], new[a + 1] = new[a + 1], new[a]
        return new

    def get_neighbor(net):
        op = random.random()
        if op < 0.18:
            return neighbor_add(net)
        elif op < 0.28:
            return neighbor_add_multi(net, random.randint(2, 3))
        elif op < 0.38:
            return neighbor_remove(net)
        elif op < 0.46:
            return neighbor_remove_multi(net, random.randint(2, 4))
        elif op < 0.56:
            return neighbor_move(net)
        elif op < 0.64:
            return neighbor_swap_adjacent(net)
        elif op < 0.74:
            return neighbor_retarget(net)
        elif op < 0.82:
            return neighbor_retarget_pair(net)
        elif op < 0.92:
            return neighbor_insert_at_depth(net)
        else:
            return neighbor_add(net)

    # Simulated annealing with periodic restarts
    T = 8.0
    T_min = 0.3
    cooling_rate = 0.9992

    restart_interval = 1500
    steps_since_restart = 0
    restart_count = 0
    max_restarts = 8

    while time.time() < deadline:
        new_net = get_neighbor(curr_net)
        new_sub_fail = verify_subset(new_net)
        new_cost = cost_of(new_net, new_sub_fail)

        delta = new_cost - curr_cost

        if delta < 0 or random.random() < np.exp(-delta / T):
            curr_net = new_net
            curr_sub_fail = new_sub_fail
            curr_cost = new_cost
            try_improve(curr_net, curr_sub_fail, curr_cost)

        T *= cooling_rate
        if T < T_min:
            T = T_min

        steps_since_restart += 1

        # Periodic restart with a perturbed version of best
        if steps_since_restart >= restart_interval and restart_count < max_restarts:
            restart_count += 1
            steps_since_restart = 0
            T = 8.0
            # Start from best with some random perturbation
            curr_net = best_net.copy()
            # Perturb: add/remove a few random comparators
            for _ in range(random.randint(2, 5)):
                if random.random() < 0.5:
                    i = random.randint(0, N - 2)
                    j = random.randint(i + 1, N - 1)
                    pos = random.randint(0, len(curr_net))
                    curr_net.insert(pos, (i, j))
                else:
                    if curr_net:
                        idx = random.randint(0, len(curr_net) - 1)
                        curr_net.pop(idx)
            curr_sub_fail = verify_subset(curr_net)
            curr_cost = cost_of(curr_net, curr_sub_fail)

    # Final full verification of best
    if best_full_fail != 0:
        ff = verify_full(best_net)
        if ff == 0:
            best_full_fail = 0
        else:
            # Fallback: return a known-valid network (odd-even) if best is invalid
            # This shouldn't happen since we only update best on full-verified valid networks
            # but guard just in case
            pass

    return best_net
