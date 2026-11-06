"""HP lattice protein folding: RL-guided growth with policy gradient + local search refinement."""
# EVOLVE-BLOCK-START
import time
import random
import math

_T0 = time.time()
_DEADLINE = 198.0  # 210s wall-clock limit, 12s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _energy_fast(seq, chain):
    """Fast energy: -count of H-H contacts on non-adjacent lattice neighbours."""
    where = {p: i for i, p in enumerate(chain)}
    h_set = set(i for i, c in enumerate(seq) if c == "H")
    contacts = 0
    for i in h_set:
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and j in h_set:
                contacts += 1
    return -contacts


class _PolicyNet:
    """Lightweight neural policy: state -> action logits for 4 directions."""
    
    def __init__(self, state_dim=8, hidden_dim=16, lr=0.05):
        self.lr = lr
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        # Initialize weights
        rng = random.Random(42)
        self.W1 = [[rng.gauss(0, 0.5) for _ in range(state_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim
        self.W2 = [[rng.gauss(0, 0.5) for _ in range(hidden_dim)] for _ in range(4)]
        self.b2 = [0.0] * 4
    
    def forward(self, state):
        """Forward pass, returns (logits, hidden_activations, state)."""
        h = []
        for j in range(self.hidden_dim):
            z = self.b1[j]
            for i in range(self.state_dim):
                z += self.W1[j][i] * state[i]
            h.append(math.tanh(z))
        
        logits = []
        for k in range(4):
            z = self.b2[k]
            for j in range(self.hidden_dim):
                z += self.W2[k][j] * h[j]
            logits.append(z)
        
        return logits, h, state
    
    def update(self, state, h, chosen_action, reward):
        """Policy gradient update: maximize expected reward."""
        logits, _, _ = self.forward(state)
        
        # Softmax
        max_logit = max(logits)
        exps = [math.exp(l - max_logit) for l in logits]
        total = sum(exps)
        probs = [e / total for e in exps]
        
        # Log probability of chosen action
        log_prob = math.log(probs[chosen_action] + 1e-10)
        
        # Gradient of log probability w.r.t. logits
        grad_logits = [0.0] * 4
        grad_logits[chosen_action] = 1.0 - probs[chosen_action]
        for k in range(4):
            if k != chosen_action:
                grad_logits[k] = -probs[k]
        
        # Backprop through W2, b2
        for k in range(4):
            self.b2[k] += self.lr * reward * grad_logits[k]
            for j in range(self.hidden_dim):
                self.W2[k][j] += self.lr * reward * grad_logits[k] * h[j]
        
        # Backprop through W1, b1
        for j in range(self.hidden_dim):
            grad_h = 0.0
            for k in range(4):
                grad_h += self.W2[k][j] * grad_logits[k]
            # tanh derivative
            dh = (1.0 - h[j] * h[j]) * grad_h
            self.b1[j] += self.lr * reward * dh
            for i in range(self.state_dim):
                self.W1[j][i] += self.lr * reward * dh * state[i]


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n < 2:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            dx = abs(chain[i][0] - chain[i - 1][0])
            dy = abs(chain[i][1] - chain[i - 1][1])
            if dx + dy != 1:
                return False
    return True


def _extract_state(sequence, chain, occupied, i):
    """
    Extract state features for the policy network at position i.
    i is the index of the residue to be placed next.
    chain contains residues 0..i-1.
    """
    if i == 0:
        x, y = chain[0]
        is_h = 1.0 if sequence[0] == 'H' else 0.0
        h_neigh = 0
        p_neigh = 0
        free = 0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (x + dx, y + dy)
            if nb not in occupied:
                free += 1
        return [is_h, float(h_neigh), float(p_neigh), float(free), 0.0, 0.0, 0.0, 0.0]
    
    # i >= 1
    # The current tail of the chain is at index i-1
    x, y = chain[i-1]
    
    if i > 1:
        px, py = chain[i-2]
        dx_prev = float(x - px)
        dy_prev = float(y - py)
    else:
        dx_prev = 0.0
        dy_prev = 0.0
    
    is_h = 1.0 if sequence[i] == 'H' else 0.0
    
    h_neigh = 0
    p_neigh = 0
    free = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = (x + dx, y + dy)
        if nb not in occupied:
            free += 1
        else:
            # The neighbor is occupied by a previously placed residue.
            # Find its type.
            for j, p in enumerate(chain):
                if p == nb:
                    if sequence[j] == 'H':
                        h_neigh += 1
                    else:
                        p_neigh += 1
                    break
    
    return [is_h, float(h_neigh), float(p_neigh), float(free), dx_prev, dy_prev, 0.0, 0.0]


def _grow_rl(sequence, dimension, rng, policy, num_episodes=10):
    """
    RL-guided growth: use policy to guide chain growth, with policy gradient updates.
    Returns a list of complete chains.
    """
    n = len(sequence)
    if n == 0:
        return [[]]
    
    h_set = set(i for i, c in enumerate(sequence) if c == "H")
    chains = []
    
    for ep in range(num_episodes):
        chain = [(0, 0)]
        occupied = {(0, 0)}
        total_reward = 0.0
        states_actions = []  # (state, action, immediate_reward)
        
        for i in range(1, n):
            x, y = chain[-1]
            
            # Get available moves
            avail_moves = []
            for idx, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
                site = (x + dx, y + dy)
                if site not in occupied:
                    avail_moves.append((idx, site))
            
            if not avail_moves:
                break
            
            # Extract state
            state = _extract_state(sequence, chain, occupied, i)
            
            # Forward pass to get logits
            logits, h, _ = policy.forward(state)
            
            # Mask unavailable actions
            masked_logits = [-1e10] * 4
            for idx, site in avail_moves:
                masked_logits[idx] = logits[idx]
            
            # Softmax over available
            max_l = max(masked_logits)
            exps = [math.exp(l - max_l) if l > -1e9 else 0.0 for l in masked_logits]
            total = sum(exps)
            if total < 1e-10:
                # All unavailable, pick first available
                chosen_idx, chosen_site = avail_moves[0]
            else:
                probs = [e / total for e in exps]
                # Sample action
                r = rng.random()
                cumsum = 0.0
                chosen_idx = avail_moves[0][0]
                chosen_site = avail_moves[0][1]
                for idx, site in avail_moves:
                    cumsum += probs[idx]
                    if r <= cumsum:
                        chosen_idx = idx
                        chosen_site = site
                        break
            
            # Compute immediate reward
            immediate_reward = 0.0
            if i in h_set:
                sx, sy = chosen_site
                for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (sx + adx, sy + ady)
                    if nb in occupied:
                        for j, p in enumerate(chain):
                            if p == nb and j < i - 1 and j in h_set:
                                immediate_reward -= 1.0  # negative = good (more H-H contact)
                                break
            
            # Small penalty for reducing free space
            free_after = 0
            sx, sy = chosen_site
            for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (sx + adx, sy + ady) not in occupied:
                    free_after += 1
            immediate_reward += 0.02 * (4 - free_after)
            
            total_reward += immediate_reward
            states_actions.append((state, h, chosen_idx, immediate_reward))
            
            # Move
            chain.append(chosen_site)
            occupied.add(chosen_site)
        
        # Update policy with accumulated reward (use final energy as reward signal)
        if len(chain) == n:
            final_energy = _energy_fast(sequence, chain)
            # Normalize reward: better (more negative) energy = higher reward
            norm_reward = -final_energy / max(1, n)
            
            for state, h, action, imm_reward in states_actions:
                # Use final reward as the signal (simplified policy gradient)
                policy.update(state, h, action, norm_reward + 0.1 * imm_reward)
            
            chains.append(list(chain))
        
        if _remaining() < 10.0:
            break
    
    if not chains:
        # Fallback: simple greedy
        chain = [(0, 0)]
        occupied = {(0, 0)}
        for i in range(1, n):
            x, y = chain[-1]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                site = (x + dx, y + dy)
                if site not in occupied:
                    chain.append(site)
                    occupied.add(site)
                    break
        chains.append(chain)
    
    return chains


def _pivot_move(chain, rng):
    """
    Pivot move: pick a cut point i (1..n-2), rotate the suffix chain[i:]
    by 90 or 180 degrees around chain[i-1].
    """
    n = len(chain)
    if n < 4:
        return None
    i = rng.randint(1, n - 2)
    px, py = chain[i - 1]
    cx, cy = chain[i]

    dx = cx - px
    dy = cy - py

    rotations = [(-dy, dx), (dy, -dx), (-dx, -dy)]
    rng.shuffle(rotations)

    for ndx, ndy in rotations:
        new_ci = (px + ndx, py + ndy)
        if new_ci in chain and new_ci != chain[i]:
            continue

        new_chain = chain[:i] + [new_ci]
        for j in range(i + 1, n):
            ox, oy = chain[j]
            nx2 = new_ci[0] + (ox - cx)
            ny2 = new_ci[1] + (oy - cy)
            new_chain.append((nx2, ny2))

        if _is_valid_chain(new_chain):
            return new_chain
    return None


def _reversal_move(chain, rng, max_len=None):
    """Random reversal move."""
    n = len(chain)
    if n < 4:
        return None
    if max_len is None:
        max_len = min(10, n - 2)
    L = rng.randint(2, max_len)
    a = rng.randint(0, n - L - 1)
    b = a + L
    new_chain = chain[:a] + chain[a:b + 1][::-1] + chain[b + 1:]
    if _is_valid_chain(new_chain):
        return new_chain
    return None


def _fragment_relocation(chain, sequence, rng, max_frag=8):
    """
    Cut out a fragment of length L and try to reinsert it at a different location
    or in a different orientation via BFS path finding.
    """
    n = len(chain)
    if n < 8:
        return None
    
    L = rng.randint(3, min(max_frag, n - 4))
    a = rng.randint(1, n - L - 2)
    b = a + L - 1
    
    fragment = chain[a:b+1]
    left_nb = chain[a-1]
    right_nb = chain[b+1]
    
    # Remove fragment
    rest = chain[:a] + chain[b+1:]
    rest_occ = set(rest)
    
    # Try to find a path of length L from left_nb to right_nb avoiding rest
    lx, ly = left_nb
    rx, ry = right_nb
    manhattan = abs(rx - lx) + abs(ry - ly)
    if manhattan > L or (L - manhattan) % 2 != 0:
        return None
    
    # BFS/DFS to find path
    def _find_path(start, end, length, occupied):
        if length == 0:
            return [start] if start == end else None
        x, y = start
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            site = (x+dx, y+dy)
            if site in occupied:
                continue
            dist = abs(end[0]-site[0]) + abs(end[1]-site[1])
            if dist > length-1 or ((length-1)-dist) % 2 != 0:
                continue
            path = _find_path(site, end, length-1, occupied | {site})
            if path is not None:
                return [site] + path
        return None
    
    path = _find_path(left_nb, right_nb, L, rest_occ)
    if path is None or len(path) != L+1 or path[-1] != right_nb:
        return None
    
    new_frag = path[1:]
    new_chain = chain[:a] + new_frag + chain[b+1:]
    if _is_valid_chain(new_chain):
        return new_chain
    return None


def _kink_move(chain, rng):
    """Kink move: move a single interior residue to an adjacent free site."""
    n = len(chain)
    if n < 3:
        return None
    i = rng.randint(1, n - 2)
    x, y = chain[i]
    occ = set(chain)
    options = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        site = (x + dx, y + dy)
        if site in occ and site != chain[i]:
            continue
        px, py = chain[i - 1]
        if abs(site[0] - px) + abs(site[1] - py) != 1:
            continue
        nx, ny = chain[i + 1]
        if abs(site[0] - nx) + abs(site[1] - ny) != 1:
            continue
        options.append(site)
    if not options:
        return None
    site = options[rng.randint(0, len(options) - 1)]
    new_chain = chain[:]
    new_chain[i] = site
    if _is_valid_chain(new_chain):
        return new_chain
    return None


def _reinsert_fragment(chain, sequence, rng, max_frag_len=6):
    """
    Cut out a fragment and try to regrow it via DFS path finding.
    """
    n = len(chain)
    if n < 6:
        return None

    L = rng.randint(2, min(max_frag_len, n - 3))
    a = rng.randint(1, n - L - 1)
    b = a + L

    left_neighbor = chain[a - 1]
    right_neighbor = chain[b]

    rest = chain[:a] + chain[b:]
    rest_occupied = set(rest)

    lx, ly = left_neighbor
    rx, ry = right_neighbor
    manhattan = abs(rx - lx) + abs(ry - ly)
    if manhattan > L or (L - manhattan) % 2 != 0:
        return None

    def _dfs(current, remaining, path, occupied):
        if remaining == 0:
            if current == right_neighbor:
                return list(path)
            return None
        x, y = current
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            dist_to_end = abs(right_neighbor[0] - site[0]) + abs(right_neighbor[1] - site[1])
            if dist_to_end > remaining or (remaining - dist_to_end) % 2 != 0:
                continue
            path.append(site)
            occupied.add(site)
            result = _dfs(site, remaining - 1, path, occupied)
            if result is not None:
                return result
            path.pop()
            occupied.discard(site)
        return None

    path = _dfs(left_neighbor, L, [left_neighbor], {left_neighbor} | rest_occupied)
    if path is None or path[-1] != right_neighbor:
        return None

    new_frag = path[1:]
    if len(new_frag) != L:
        return None

    new_chain = chain[:a] + new_frag + chain[b:]
    if _is_valid_chain(new_chain):
        return new_chain
    return None


def _simulated_annealing(sequence, chain, rng, t_start=1.0, t_end=0.01, max_iters=3000):
    """
    Simulated annealing with pivot, reversal, kink, reinsert, and fragment relocation moves.
    Returns (best_chain, best_energy).
    """
    best_chain = list(chain)
    best_energy = _energy_fast(sequence, best_chain)
    cur_chain = list(chain)
    cur_energy = best_energy
    n = len(cur_chain)

    # Adaptive move weights: pivot, reversal, kink, reinsert, fragment_relocation
    move_weights = [1.0, 1.0, 0.8, 0.5, 0.7]
    move_success = [0, 0, 0, 0, 0]
    move_attempts = [0, 0, 0, 0, 0]

    total_steps = max(100, max_iters)
    for it in range(total_steps):
        if _remaining() < 5.0:
            break

        if total_steps > 1:
            frac = it / total_steps
            T = t_start * (t_end / t_start) ** frac
        else:
            T = t_start

        # Choose move type
        r = rng.random()
        cumulative = [0]
        for w in move_weights:
            cumulative.append(cumulative[-1] + w)
        total_w = cumulative[-1]
        r *= total_w
        if r < cumulative[1]:
            move_type = 0  # pivot
        elif r < cumulative[2]:
            move_type = 1  # reversal
        elif r < cumulative[3]:
            move_type = 2  # kink
        elif r < cumulative[4]:
            move_type = 3  # reinsert
        else:
            move_type = 4  # fragment relocation

        move_attempts[move_type] += 1

        if move_type == 0:
            new_chain = _pivot_move(cur_chain, rng)
        elif move_type == 1:
            new_chain = _reversal_move(cur_chain, rng)
        elif move_type == 2:
            new_chain = _kink_move(cur_chain, rng)
        elif move_type == 3:
            new_chain = _reinsert_fragment(cur_chain, sequence, rng)
        else:
            new_chain = _fragment_relocation(cur_chain, sequence, rng)

        if new_chain is None:
            continue

        new_energy = _energy_fast(sequence, new_chain)
        delta = new_energy - cur_energy

        if delta <= 0:
            cur_chain = new_chain
            cur_energy = new_energy
            move_success[move_type] += 1
            if cur_energy < best_energy:
                best_chain = list(cur_chain)
                best_energy = cur_energy
        else:
            if T > 0 and rng.random() < math.exp(-delta / T):
                cur_chain = new_chain
                cur_energy = new_energy
                move_success[move_type] += 1

        # Adapt move weights every 150 iterations
        if it % 150 == 149:
            for k in range(5):
                if move_attempts[k] > 0:
                    rate = move_success[k] / move_attempts[k]
                    move_weights[k] = max(0.1, rate * 2.5)
                move_attempts[k] = 0
                move_success[k] = 0

    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using RL-guided growth + SA refinement with time guard."""
    rng = random.Random(777 + len(sequence) * 13)
    n = len(sequence)

    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    # Create policy network
    policy = _PolicyNet(state_dim=8, hidden_dim=16, lr=0.05)

    # Generate initial chains via RL-guided growth
    rl_chains = _grow_rl(sequence, dimension, rng, policy, num_episodes=8)

    best_chain = None
    best_energy = 0

    # Evaluate all initial chains
    for ch in rl_chains:
        if len(ch) == n and _is_valid_chain(ch):
            e = _energy_fast(sequence, ch)
            if e < best_energy:
                best_energy = e
                best_chain = list(ch)

    # Fallback if RL didn't produce valid chains
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy_fast(sequence, best_chain)

    # Build SA starting points
    sa_runs = [best_chain]
    for ch in rl_chains:
        if len(ch) == n and _is_valid_chain(ch) and ch != best_chain:
            sa_runs.append(ch)
        if len(sa_runs) >= 5:
            break

    # If very little time, do quick SA
    if _remaining() < 15.0:
        if _remaining() > 5.0:
            improved, imp_e = _simulated_annealing(
                sequence, best_chain, rng,
                t_start=0.7, t_end=0.02, max_iters=800
            )
            if imp_e < best_energy:
                best_energy = imp_e
                best_chain = improved
        return [list(p) for p in best_chain]

    # Distribute time among SA runs
    num_sa = len(sa_runs)
    time_per_sa = (_remaining() - 12.0) / max(1, num_sa)

    for idx, start_chain in enumerate(sa_runs):
        if _remaining() < 10.0:
            break

        iters = int(time_per_sa * 60)
        iters = max(400, min(4000, iters))

        t_start = 1.0 if idx == 0 else 0.6
        t_end = 0.01

        improved, imp_e = _simulated_annealing(
            sequence, start_chain, rng,
            t_start=t_start, t_end=t_end, max_iters=iters
        )
        if imp_e < best_energy:
            best_energy = imp_e
            best_chain = improved

    # Final polish
    if best_chain is not None and _remaining() > 10.0:
        iters = int((_remaining() - 6.0) * 50)
        iters = max(400, min(2500, iters))
        improved, imp_e = _simulated_annealing(
            sequence, best_chain, rng,
            t_start=0.3, t_end=0.005, max_iters=iters
        )
        if imp_e < best_energy:
            best_energy = imp_e
            best_chain = improved

    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
