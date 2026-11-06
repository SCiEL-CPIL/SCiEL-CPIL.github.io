"""Size-efficient sorting networks on 17 wires – improved search."""
import time
import random
import numpy as np

N = 17
MAX_COMP = 4096

# ---------- fast verification helpers ----------
def _make_valid_mask():
    """Boolean array of length 2**N where entry k is True iff k is a sorted 0/1 state."""
    m = np.zeros(1 << N, dtype=bool)
    for k in range(N + 1):
        if k == 0:
            val = 0
        else:
            val = (1 << N) - (1 << (N - k))
        m[val] = True
    return m

_VALID = _make_valid_mask()

# Pre-compute wire masks for faster extraction
_WIRE_MASKS = [(1 << i) for i in range(N)]

def verify(network):
    """Return True if network sorts all 2**N boolean inputs (zero-one principle)."""
    if not network:
        return False
    
    states = np.arange(1 << N, dtype=np.uint32)
    for i, j in network:
        # Extract bits i and j from all states
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        # Where wire i has 1 and wire j has 0, swap them
        mask = (bi == 1) & (bj == 0)
        if np.any(mask):
            states[mask] ^= _WIRE_MASKS[i] | _WIRE_MASKS[j]
    return bool(np.all(_VALID[states]))

def compute_depth(network):
    """Greedy layering depth (reported separately; does not affect score)."""
    if not network:
        return 0
    layers = []
    for comp in network:
        placed = False
        for layer in layers:
            wires = set()
            for c in layer:
                wires.add(c[0]); wires.add(c[1])
            if comp[0] not in wires and comp[1] not in wires:
                layer.append(comp)
                placed = True
                break
        if not placed:
            layers.append([comp])
    return len(layers)

def seed_network():
    """Odd-even transposition network (136 comparators for N=17)."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

# ---------- search helpers ----------
def try_remove(network, idx):
    """Return new network if removing comparator idx keeps validity, else None."""
    new_net = network[:idx] + network[idx + 1:]
    if verify(new_net):
        return new_net
    return None

def greedy_remove_until_fixed(network, deadline, rng, verbose=False):
    """Repeatedly try to remove any comparator (random order each pass) until stuck."""
    current = list(network)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(current)))
        rng.shuffle(indices)
        for idx in indices:
            if time.time() >= deadline:
                break
            reduced = try_remove(current, idx)
            if reduced is not None:
                current = reduced
                improved = True
                if verbose:
                    print(f"  removed one -> {len(current)}", flush=True)
                break
    return current

def local_search(network, deadline, rng, verbose=False):
    """
    Local search: try random replacements and reorders, then greedy remove.
    """
    current = list(network)
    best = list(current)
    best_count = len(current)
    
    while time.time() < deadline:
        # Try to remove
        current = greedy_remove_until_fixed(current, deadline, rng)
        if len(current) < best_count:
            best = list(current)
            best_count = len(current)
            if verbose:
                print(f"  improved to {best_count}", flush=True)
        
        if time.time() >= deadline:
            break
            
        # If we are stuck (greedy didn't improve), try a move
        # Move 1: Swap two comparators
        if len(current) >= 2 and rng.random() < 0.5:
            i, j = rng.sample(range(len(current)), 2)
            new_net = list(current)
            new_net[i], new_net[j] = new_net[j], new_net[i]
            if verify(new_net):
                current = new_net
                continue # Go back to greedy remove
                
        # Move 2: Replace one comparator
        elif current:
            idx = rng.randint(0, len(current) - 1)
            a, b = rng.sample(range(N), 2)
            if a > b:
                a, b = b, a
            new_net = current[:idx] + [(a, b)] + current[idx + 1:]
            if verify(new_net):
                current = new_net
                continue # Go back to greedy remove
                
        # Move 3: Insert a random comparator (to grow, then shrink?)
        # This is risky, might get stuck in large networks. 
        # But sometimes helps escape local minima.
        else:
            a, b = rng.sample(range(N), 2)
            if a > b:
                a, b = b, a
            pos = rng.randint(0, len(current))
            new_net = current[:pos] + [(a, b)] + current[pos:]
            if verify(new_net):
                current = new_net
                continue

    return best

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s hard limit
    rng = random.Random(42)

    best_net = seed_network()
    best_count = len(best_net)
    print(f"Seed network: {best_count} comparators", flush=True)

    # Ensure seed is valid
    if not verify(best_net):
        best_net = []
        best_count = 0

    # Phase 1: Aggressive local search from seed
    print("Phase 1: Local search from seed...", flush=True)
    best_net = local_search(best_net, deadline, rng, verbose=True)
    best_count = len(best_net)
    print(f"Phase 1 done: {best_count}", flush=True)

    # Phase 2: Multiple restarts with perturbations
    print("Phase 2: Restarting with perturbations...", flush=True)
    for restart in range(100):
        if time.time() >= deadline:
            break
        
        # Create a perturbed version of the best network
        cand = list(best_net)
        
        # Perturb: swap a few, replace a few
        for _ in range(5):
            if not cand:
                break
            move = rng.random()
            if move < 0.5 and len(cand) >= 2:
                i, j = rng.sample(range(len(cand)), 2)
                cand[i], cand[j] = cand[j], cand[i]
            else:
                idx = rng.randint(0, len(cand) - 1)
                a, b = rng.sample(range(N), 2)
                if a > b:
                    a, b = b, a
                cand[idx] = (a, b)
        
        # Run local search on the perturbed network
        cand = local_search(cand, deadline, rng, verbose=False)
        
        if len(cand) < best_count:
            best_net = cand
            best_count = len(cand)
            print(f"  Restart {restart}: improved to {best_count}", flush=True)

    # Final verification
    if not verify(best_net):
        best_net = seed_network()
        best_count = len(best_net)

    depth = compute_depth(best_net)
    print(f"Final: {best_count} comparators, depth {depth}", flush=True)
    return best_net
