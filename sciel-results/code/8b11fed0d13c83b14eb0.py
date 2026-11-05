"""Search for a minimal sorting network on 17 wires using simulated annealing."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136  # odd-even transposition size
DEADLINE = 75.0  # safety margin before 90s kill


def _apply_comparator(state, i, j):
    """Apply compare-exchange: min to i, max to j. state is a list/array."""
    a, b = state[i], state[j]
    if a > b:
        state[i], state[j] = b, a
    return state


def _is_sorted(state):
    for k in range(N - 1):
        if state[k] > state[k + 1]:
            return False
    return True


def _verify(network):
    """Verify network sorts all 2^17 boolean inputs (0-1 principle)."""
    # Represent network as list of (i, j)
    # For each of 2^17 inputs, check sortedness
    # Use vectorized-ish approach: track which inputs are still "live" (not yet sorted)
    # Actually, just iterate all inputs but prune early.
    
    n_inputs = 1 << N
    # We'll process inputs in batches for speed, but correctness is paramount.
    # A faster approach: simulate all inputs simultaneously using bitmasks.
    # For each wire, maintain a bitmask of which inputs have value 1 on that wire.
    # Initially, wire k has value 1 in input x iff bit k of x is 1.
    
    # bitmask[w] = integer where bit x is 1 iff input x has value 1 on wire w
    # For N=17, each bitmask is a 17-bit integer (2^17 values)
    
    masks = [0] * N
    for w in range(N):
        # Inputs where bit w is set: all x where (x >> w) & 1 == 1
        # That's a pattern of 2^w ones, 2^w zeros, repeating
        block = 1 << w
        pattern = 0
        for x in range(n_inputs):
            if (x >> w) & 1:
                pattern |= (1 << x)
        masks[w] = pattern
    
    # Apply each comparator
    for i, j in network:
        a = masks[i]
        b = masks[j]
        # min goes to i: bit x of min = 1 iff a_x AND b_x
        # max goes to j: bit x of max = 1 iff a_x OR b_x
        masks[i] = a & b
        masks[j] = a | b
    
    # Check sortedness: for each adjacent pair (k, k+1),
    # we need masks[k] & ~masks[k+1] == 0 for all inputs
    # i.e., no input has wire k = 1 and wire k+1 = 0
    for k in range(N - 1):
        if masks[k] & ~masks[k + 1] != 0:
            return False
    return True


def _depth(network):
    """Compute parallel depth of the network."""
    wire_depth = [0] * N
    for i, j in network:
        d = max(wire_depth[i], wire_depth[j]) + 1
        wire_depth[i] = d
        wire_depth[j] = d
    return max(wire_depth) if wire_depth else 0


def _odd_even_transposition():
    """Seed: odd-even transposition network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _random_network(n_comp):
    """Generate a random network with n_comp comparators."""
    return [(random.randint(0, N - 2), random.randint(1, N - 1))
            for _ in range(n_comp)]


def _swap_in_network(network, idx):
    """Return a new network with comparator at idx swapped with another random position."""
    new_net = list(network)
    j = random.randint(0, len(new_net) - 1)
    new_net[idx], new_net[j] = new_net[j], new_net[idx]
    return new_net


def _mutate_comparator(network, idx):
    """Return a new network with comparator at idx changed to a random pair."""
    new_net = list(network)
    i = random.randint(0, N - 2)
    j = random.randint(i + 1, N - 1)
    new_net[idx] = (i, j)
    return new_net


def _insert_comparator(network):
    """Return a new network with one random comparator inserted."""
    new_net = list(network)
    i = random.randint(0, N - 2)
    j = random.randint(i + 1, N - 1)
    pos = random.randint(0, len(new_net))
    new_net.insert(pos, (i, j))
    return new_net


def _delete_comparator(network):
    """Return a new network with one random comparator deleted."""
    if len(network) <= 1:
        return list(network)
    new_net = list(network)
    pos = random.randint(0, len(new_net) - 1)
    new_net.pop(pos)
    return new_net


def construct():
    """Construct a sorting network on 17 wires with as few comparators as possible."""
    start_time = time.time()
    random.seed(42)
    
    # Start from the seed odd-even transposition network
    best_net = _odd_even_transposition()
    best_size = len(best_net)
    assert best_size == SEED_COMPARATORS, f"Expected {SEED_COMPARATORS}, got {best_size}"
    
    # Verify seed is correct
    if not _verify(best_net):
        raise ValueError("Seed network is not a valid sorting network")
    
    # Simulated annealing
    current_net = list(best_net)
    current_size = len(current_net)
    
    # Temperature schedule
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    # Precompute for faster verification: we'll verify fully each time
    # but that's 2^17 * comparators operations. Let's estimate:
    # _verify does N initializations + len(network) * 2 mask ops + N-1 checks
    # Each mask op is on Python ints (up to 2^17 bits), which is fast.
    
    iterations = 0
    max_iterations = 200000  # safety cap
    
    while time.time() - start_time < DEADLINE and iterations < max_iterations:
        iterations += 1
        
        # Choose a mutation
        r = random.random()
        if current_size > 63 and r < 0.3:
            # Try to delete a comparator
            new_net = _delete_comparator(current_net)
            delta_size = -1
        elif r < 0.5:
            # Mutate a random comparator
            idx = random.randint(0, len(current_net) - 1)
            new_net = _mutate_comparator(current_net, idx)
            delta_size = 0
        elif r < 0.7:
            # Swap two comparators
            idx = random.randint(0, len(current_net) - 1)
            new_net = _swap_in_network(current_net, idx)
            delta_size = 0
        elif r < 0.85:
            # Insert a comparator
            new_net = _insert_comparator(current_net)
            delta_size = 1
        else:
            # Reorder: move a comparator to a random position
            idx = random.randint(0, len(current_net) - 1)
            comp = current_net[idx]
            new_net = list(current_net)
            new_net.pop(idx)
            pos = random.randint(0, len(new_net))
            new_net.insert(pos, comp)
            delta_size = 0
        
        # Verify the new network
        if _verify(new_net):
            new_size = len(new_net)
            # Accept if better or with some probability (annealing)
            if new_size < current_size:
                current_net = new_net
                current_size = new_size
                if new_size < best_size:
                    best_net = new_net
                    best_size = new_size
                    # Cool down more aggressively when we find improvement
                    T = max(T_min, T * 0.9)
            elif new_size == current_size:
                # Accept with probability exp(-delta/T) where delta=0, so always accept
                # but only if it doesn't hurt depth too much
                current_net = new_net
                current_size = new_size
            else:
                # Worse size, accept with probability exp(-delta_size/T)
                if random.random() < np.exp(-delta_size / T):
                    current_net = new_net
                    current_size = new_size
        else:
            # Invalid network, reject
            pass
        
        # Cool down
        T = max(T_min, T * cooling)
        
        # If we've been stuck for a while, restart from best
        # (implicit via the acceptance criteria)
    
    # Final verification
    if not _verify(best_net):
        # Fallback to seed
        best_net = _odd_even_transposition()
    
    return best_net


# Quick self-test if run directly
if __name__ == "__main__":
    net = construct()
    print(f"Network size: {len(net)}")
    print(f"Depth: {_depth(net)}")
    print(f"Valid: {_verify(net)}")
