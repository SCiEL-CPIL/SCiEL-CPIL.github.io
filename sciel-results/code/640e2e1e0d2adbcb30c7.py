import time
import random
import math
import numpy as np

_BEST = None
_DEADLINE = 0.0

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _fast_greedy(k, n_rows, rng=None, spread=False):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows

class PPOPolicy:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=1e-3, gamma=0.99, epsilon=1e-8):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Simple MLP: Input -> Hidden -> Output
        # Weights
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, action_dim) * 0.01
        self.b2 = np.zeros(action_dim)
        
        # Value Head
        self.Wv = np.random.randn(hidden_dim, 1) * 0.01
        self.bv = np.zeros(1)

    def forward(self, state):
        # state: (batch, state_dim)
        h = np.tanh(state @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        values = (h @ self.Wv + self.bv).flatten()
        return logits, values

    def get_action(self, state):
        # state: (state_dim,)
        state_batch = state.reshape(1, -1)
        logits, _ = self.forward(state_batch)
        logits = logits[0]
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        action = np.random.choice(len(probs), p=probs)
        log_prob = np.log(probs[action] + self.epsilon)
        return action, log_prob

    def train(self, states, actions, log_probs, rewards, values):
        # Simple policy gradient update (REINFORCE style approximation for simplicity and speed)
        # In a real PPO, we'd compute advantages and clip. Here we do a basic gradient step.
        batch_size = len(states)
        if batch_size == 0: return
        
        # Normalize rewards for stability
        rewards = np.array(rewards)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-6)
        
        # Compute loss: - sum(log_prob * reward)
        # We need gradients. Since we don't have autograd, we use finite differences or a simple manual backprop?
        # Manual backprop for this small net is feasible but tedious.
        # Alternative: Use a very simple heuristic update or just stick to greedy if RL is too slow to implement correctly in pure numpy without autograd.
        
        # Given the 120s limit and complexity, implementing full PPO backprop in pure numpy is risky.
        # Let's switch to a **Neural Bandit / Contextual Bandit** approach which is simpler:
        # We just update weights in the direction of the action taken if reward is positive.
        # Or better: Use the existing SA but initialized by a **Genetic Algorithm** that evolves row patterns?
        
        # Actually, let's pivot the "RL" idea to a **Neural Network Guided Local Search**.
        # The "RL" aspect is that we train a simple model to predict good "gaps" based on the current state.
        # But without autograd, training is hard.
        
        # Let's stick to a **Differentiable Relaxation** idea? No, integers.
        
        # Let's try **Simulated Annealing with a Neural Network-based Proposal Distribution**.
        # We can't train the NN easily. 
        
        # Let's reconsider the "Multi-Agent" aspect.
        # What if we use **Hyper-heuristics**? 
        # A meta-heuristic that chooses between different low-level heuristics (greedy, random, swap) based on the current state of the board.
        # This is structurally different from fixed SA.
        
        # Let's implement a **Hyper-heuristic** that uses a simple Markov Chain to switch between move types, 
        # and the transition probabilities are adapted based on recent success/failure (a form of RL without NN weights, just probability updates).
        
        pass # Placeholder, we will implement the Hyper-heuristic in the main construct function logic instead of this class if it's too complex.

def _hyper_heuristic_optimize(rows, k, iters, rng, deadline):
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    # State for the hyper-heuristic: probabilities of choosing each move type
    # Move types: 0: Single Element Move, 1: Row Rebuild, 2: Scope Compression
    p_moves = [0.5, 0.3, 0.2]
    last_rewards = [0.0, 0.0, 0.0]
    
    def get_cost(rows):
        # Simple cost: scope + collisions
        seen = set()
        collisions = 0
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in seen:
                        collisions += 1
                    else:
                        seen.add(d)
        return _scope(rows) + 10 * collisions

    current_cost = get_cost(current_rows)
    best_cost = current_cost
    
    T = 5.0
    cooling = 0.999
    
    for it in range(iters):
        if time.time() > deadline:
            break
            
        # Select move type based on probabilities
        move_type = rng.choices([0, 1, 2], weights=p_moves)[0]
        
        old_cost = current_cost
        old_rows_copy = [list(r) for r in current_rows]
        
        if move_type == 0: # Single Element Move
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            old_val = current_rows[i][j]
            prev_val = current_rows[i][j-1]
            next_val = current_rows[i][j+1] if j < k else current_scope + 10
            
            lo = prev_val + 1
            hi = next_val - 1
            if lo > hi: continue
            
            # Random perturbation
            new_val = rng.randint(lo, hi)
            if new_val == old_val: continue
            
            current_rows[i][j] = new_val
            new_cost = get_cost(current_rows)
            
        elif move_type == 1: # Row Rebuild (Greedy with randomization)
            i = rng.randint(0, n_rows - 1)
            # Remove row i
            rows_without_i = [r for idx, r in enumerate(current_rows) if idx != i]
            
            # Rebuild row i greedily trying to fit in gaps
            new_row = [0]
            x = 0
            used_diffs = set()
            for r in rows_without_i:
                for a in r:
                    for b in r:
                        if a != b:
                            used_diffs.add(abs(a-b))
            
            for _ in range(k):
                found = False
                # Try small steps first
                for step in range(1, 20):
                    t = x + step
                    new_diffs = [t - a for a in new_row]
                    if all(d not in used_diffs for d in new_diffs):
                        new_row.append(t)
                        used_diffs.update(new_diffs)
                        x = t
                        found = True
                        break
                if not found:
                    # Fallback: find any valid t
                    t = x + 1
                    while True:
                        new_diffs = [t - a for a in new_row]
                        if all(d not in used_diffs for d in new_diffs):
                            new_row.append(t)
                            used_diffs.update(new_diffs)
                            x = t
                            break
                        t += 1
            current_rows[i] = new_row
            new_cost = get_cost(current_rows)
            
        else: # Scope Compression
            # Find max element and try to reduce it
            max_val = current_scope
            max_row_idx = -1
            max_pos_idx = -1
            for ri in range(n_rows):
                for pi in range(k, 0, -1):
                    if current_rows[ri][pi] == max_val:
                        max_row_idx = ri
                        max_pos_idx = pi
                        break
                if max_row_idx >= 0: break
            
            if max_row_idx < 0: continue
            
            i = max_row_idx
            j = max_pos_idx
            old_val = current_rows[i][j]
            prev_val = current_rows[i][j-1]
            
            new_val = rng.randint(prev_val + 1, old_val - 1)
            if new_val <= prev_val or new_val >= old_val: continue
            
            current_rows[i][j] = new_val
            new_cost = get_cost(current_rows)
            
        # Acceptance
        delta = new_cost - old_cost
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.01)):
            current_cost = new_cost
            reward = -delta # Improvement is positive reward
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_scope = _scope(current_rows)
                best_rows = [list(r) for r in current_rows]
                
            # Update Hyper-heuristic probabilities
            # Simple update: increase prob of successful move, decrease others
            # Use a learning rate
            lr = 0.1
            for idx in range(3):
                if idx == move_type:
                    p_moves[idx] += lr * reward
                else:
                    p_moves[idx] -= lr * reward * 0.5
                    
            # Normalize and clip
            p_moves = [max(0.01, p) for p in p_moves]
            total = sum(p_moves)
            p_moves = [p/total for p in p_moves]
            
        else:
            # Revert
            current_rows = old_rows_copy
            current_cost = old_cost
            # Penalize the move
            p_moves[move_type] = max(0.01, p_moves[move_type] - 0.05)
            total = sum(p_moves)
            p_moves = [p/total for p in p_moves]
            
        T = max(T * cooling, 0.01)
        
    return best_scope, best_rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 108.0
    
    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])
            
    rng = random.Random(42)
    
    # 1. Generate initial candidates
    candidates = []
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    
    # Randomized greedy
    for _ in range(4):
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                max_step = rng.randint(1, 8)
                for t in range(x + 1, x + 1 + max_step * 3):
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    if (used & new) == 0:
                        row.append(t)
                        used |= new
                        x = t
                        found = True
                        break
                if not found:
                    t = x + 1
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    while used & new:
                        t += 1
                        new = 0
                        for a in row:
                            new |= (1 << (t - a))
                    row.append(t)
                    used |= new
                    x = t
            rows.append(row)
        candidates.append(rows)
        
    candidates.sort(key=_scope)
    
    # 2. Run Hyper-heuristic optimization on top candidates
    # This is the "RL-like" structural change: adaptive move selection
    for idx, cand in enumerate(candidates[:3]):
        if time.time() > _DEADLINE:
            break
        remaining = _DEADLINE - time.time()
        # Allocate time based on rank
        if idx == 0:
            hh_deadline = time.time() + remaining * 0.4
        elif idx == 1:
            hh_deadline = time.time() + remaining * 0.3
        else:
            hh_deadline = _DEADLINE
            
        iters = int((hh_deadline - time.time()) * 1000) # Adjust speed factor
        iters = max(iters, 1000)
        
        s, rows = _hyper_heuristic_optimize(cand, k, iters, rng, hh_deadline)
        consider(rows)
        
    # 3. Polish with standard SA if time remains
    if _BEST is not None and time.time() < _DEADLINE:
        remaining_time = _DEADLINE - time.time()
        if remaining_time > 5:
            # Simple SA polish
            s, rows = _sa_optimize_polish(_BEST[1], k, int(remaining_time * 5000), rng, _DEADLINE)
            consider(rows)
            
    if _BEST is not None:
        return _BEST[1]
    return candidates[0]

def _sa_optimize_polish(rows, k, iters, rng, deadline):
    # Lightweight SA for final polish
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    def get_cost(rows):
        seen = set()
        collisions = 0
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in seen:
                        collisions += 1
                    else:
                        seen.add(d)
        return _scope(rows) + 10 * collisions

    current_cost = get_cost(current_rows)
    best_cost = current_cost
    T = 0.5
    cooling = 0.999
    
    for it in range(iters):
        if time.time() > deadline: break
        
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        old_val = current_rows[i][j]
        prev_val = current_rows[i][j-1]
        next_val = current_rows[i][j+1] if j < k else current_scope + 10
        
        lo = prev_val + 1
        hi = next_val - 1
        if lo > hi: continue
        
        new_val = rng.randint(lo, hi)
        if new_val == old_val: continue
        
        current_rows[i][j] = new_val
        new_cost = get_cost(current_rows)
        
        delta = new_cost - current_cost
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.01)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_scope = _scope(current_rows)
                best_rows = [list(r) for r in current_rows]
        else:
            current_rows[i][j] = old_val
            
        T = max(T * cooling, 0.01)
        
    return best_scope, best_rows
