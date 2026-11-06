import time
import random
import math
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560

# Precompute triple index lookup
_triple_idx = {}
for _i, _t in enumerate(ALL_TRIPLES):
    _triple_idx[_t] = _i

def _tidx(a, b, c):
    """Get index of triple {a,b,c} (all distinct)."""
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return _triple_idx[(a, b, c)]

class Hypergraph:
    def __init__(self):
        self.presence = 0  # 560-bit integer
        self.common = [[0] * V for _ in range(V)]  # 16x16 matrix of 16-bit masks
        self.count = 0

    def add(self, idx):
        a, b, c = ALL_TRIPLES[idx]
        self.presence |= (1 << idx)
        self.count += 1
        self.common[a][b] |= (1 << c)
        self.common[b][a] |= (1 << c)
        self.common[a][c] |= (1 << b)
        self.common[c][a] |= (1 << b)
        self.common[b][c] |= (1 << a)
        self.common[c][b] |= (1 << a)

    def remove(self, idx):
        a, b, c = ALL_TRIPLES[idx]
        self.presence &= ~(1 << idx)
        self.count -= 1
        self.common[a][b] &= ~(1 << c)
        self.common[b][a] &= ~(1 << c)
        self.common[a][c] &= ~(1 << b)
        self.common[c][a] &= ~(1 << b)
        self.common[b][c] &= ~(1 << a)
        self.common[c][b] &= ~(1 << a)

    def is_present(self, idx):
        return (self.presence >> idx) & 1

    def check_conflict(self, a, b, c):
        """
        Check if adding {a,b,c} creates a K(2,2,2).
        We assume {a,b,c} is not currently present.
        Returns (bool, list_of_conflicting_triple_indices) for weight updates.
        """
        mask_bc = self.common[b][c] & ~(1 << a) & ~(1 << b) & ~(1 << c)
        mask_ac = self.common[a][c] & ~(1 << a) & ~(1 << b) & ~(1 << c)
        mask_ab = self.common[a][b] & ~(1 << a) & ~(1 << b) & ~(1 << c)

        conflicting = set()

        a_p_mask = mask_bc
        while a_p_mask:
            a_p_bit = a_p_mask & (-a_p_mask)
            a_p = a_p_bit.bit_length() - 1
            a_p_mask ^= a_p_bit
            
            b_p_mask = mask_ac & ~(1 << a_p)
            while b_p_mask:
                b_p_bit = b_p_mask & (-b_p_mask)
                b_p = b_p_bit.bit_length() - 1
                b_p_mask ^= b_p_bit
                
                if not (self.common[a_p][b_p] & (1 << c)):
                    continue
                
                c_cand = (self.common[a][b] & 
                          self.common[a][b_p] & 
                          self.common[a_p][b] & 
                          self.common[a_p][b_p])
                c_cand &= ~(1 << a)
                c_cand &= ~(1 << b)
                c_cand &= ~(1 << c)
                c_cand &= ~(1 << a_p)
                c_cand &= ~(1 << b_p)
                
                if c_cand:
                    # Found conflict - identify which triples are involved
                    # The new triple {a,b,c} conflicts with:
                    # {a,b,c'}, {a',b,c'}, {a,b',c'}, {a',b',c'} for some c'
                    # Also {a',b',c}, {a',b,c}, {a,b',c}
                    # We mark all these as conflicting
                    cc = c_cand
                    while cc:
                        c_bit = cc & (-cc)
                        c_val = c_bit.bit_length() - 1
                        cc ^= c_bit
                        # {a,b,c'}
                        try:
                            conflicting.add(_tidx(a, b, c_val))
                        except: pass
                        # {a',b,c'}
                        try:
                            conflicting.add(_tidx(a_p, b, c_val))
                        except: pass
                        # {a,b',c'}
                        try:
                            conflicting.add(_tidx(a, b_p, c_val))
                        except: pass
                        # {a',b',c'}
                        try:
                            conflicting.add(_tidx(a_p, b_p, c_val))
                        except: pass
                    # Also the other triples in the K(2,2,2)
                    try:
                        conflicting.add(_tidx(a_p, b_p, c))
                    except: pass
                    try:
                        conflicting.add(_tidx(a_p, b, c))
                    except: pass
                    try:
                        conflicting.add(_tidx(a, b_p, c))
                    except: pass
                    return True, list(conflicting)
        
        return False, []

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin
    
    best_hg = None
    best_count = 0
    
    # Dynamic weights for vertex selection
    weights = [1.0] * N_TRIPLES
    
    restarts = 0
    max_restarts = 20
    
    while time.time() < deadline - 10 and restarts < max_restarts:
        restarts += 1
        
        # Weighted greedy initialization
        hg = Hypergraph()
        remaining = list(range(N_TRIPLES))
        
        # Sort by weight with some randomness
        for step in range(len(remaining)):
            if time.time() > deadline - 15:
                break
            # Pick from top-weighted candidates with some randomness
            # Use exponential sampling based on weights
            n_remain = len(remaining)
            if n_remain <= 10:
                candidates = remaining
            else:
                # Sample a subset of high-weight candidates
                idxs = list(range(n_remain))
                # Sort by weight (descending) and take top portion
                idxs.sort(key=lambda i: weights[remaining[i]], reverse=True)
                top_k = min(n_remain, max(20, n_remain // 3))
                candidates = [remaining[i] for i in idxs[:top_k]]
            
            # Random choice from candidates
            idx = random.choice(candidates)
            a, b, c = ALL_TRIPLES[idx]
            
            if not hg.check_conflict(a, b, c)[0]:
                hg.add(idx)
            else:
                # Update weights: penalize conflicting triples
                _, conflicts = hg.check_conflict(a, b, c)
                for cf in conflicts:
                    weights[cf] *= 0.95
                weights[idx] *= 0.98
            
            remaining.remove(idx)
        
        # Simulated Annealing local search
        present_mask = hg.presence
        present_list = [i for i in range(N_TRIPLES) if (present_mask >> i) & 1]
        missing = [i for i in range(N_TRIPLES) if not (present_mask >> i) & 1]
        
        # SA parameters
        T_start = 2.0
        T_end = 0.05
        cooling = 0.995
        T = T_start
        sa_steps = 0
        max_sa_steps = 5000
        
        current_count = hg.count
        best_local_count = current_count
        best_local_presence = hg.presence
        best_local_common = [row[:] for row in hg.common]
        
        while T > T_end and sa_steps < max_sa_steps and time.time() < deadline - 8:
            sa_steps += 1
            
            if not missing:
                # No missing triples, try swaps
                if present_list:
                    idx1 = random.choice(present_list)
                    # Remove it and see if we can add something
                    hg.remove(idx1)
                    new_count = hg.count
                    # Try to add a random missing triple
                    if missing:
                        idx2 = random.choice(missing)
                        a2, b2, c2 = ALL_TRIPLES[idx2]
                        conflict, _ = hg.check_conflict(a2, b2, c2)
                        if not conflict:
                            hg.add(idx2)
                            new_count = hg.count
                            delta = new_count - current_count
                            if delta > 0 or random.random() < math.exp(-delta / T):
                                current_count = new_count
                                present_list.remove(idx1)
                                present_list.append(idx2)
                                missing.remove(idx2)
                                if current_count > best_local_count:
                                    best_local_count = current_count
                                    best_local_presence = hg.presence
                                    best_local_common = [row[:] for row in hg.common]
                            else:
                                # Revert
                                if not conflict:
                                    hg.remove(idx2)
                                hg.add(idx1)
                        else:
                            hg.add(idx1)
                    else:
                        hg.add(idx1)
                T *= cooling
                continue
            
            # Choose a move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.4 and missing:
                # Try to add a missing triple
                idx = random.choice(missing)
                a, b, c = ALL_TRIPLES[idx]
                conflict, conflicts = hg.check_conflict(a, b, c)
                if not conflict:
                    hg.add(idx)
                    new_count = hg.count
                    delta = new_count - current_count
                    current_count = new_count
                    missing.remove(idx)
                    present_list.append(idx)
                    if current_count > best_local_count:
                        best_local_count = current_count
                        best_local_presence = hg.presence
                        best_local_common = [row[:] for row in hg.common]
                    # Boost weight of this triple
                    weights[idx] *= 1.05
                else:
                    # Penalize conflicting triples
                    for cf in conflicts:
                        weights[cf] *= 0.97
                    weights[idx] *= 0.99
                    
            elif move_type < 0.7 and present_list:
                # Try to remove a present triple (to allow additions later)
                idx = random.choice(present_list)
                hg.remove(idx)
                new_count = hg.count
                delta = new_count - current_count
                # Accept if delta >= 0 or with probability exp(delta/T)
                if delta >= 0 or random.random() < math.exp(delta / T):
                    current_count = new_count
                    present_list.remove(idx)
                    missing.append(idx)
                    weights[idx] *= 1.02  # Slight boost since removal might help
                else:
                    hg.add(idx)
                    
            else:
                # Swap: remove one present, try to add one missing
                if present_list and missing:
                    idx1 = random.choice(present_list)
                    idx2 = random.choice(missing)
                    hg.remove(idx1)
                    a2, b2, c2 = ALL_TRIPLES[idx2]
                    conflict, _ = hg.check_conflict(a2, b2, c2)
                    if not conflict:
                        hg.add(idx2)
                        new_count = hg.count
                        delta = new_count - current_count
                        if delta >= 0 or random.random() < math.exp(delta / T):
                            current_count = new_count
                            present_list.remove(idx1)
                            present_list.append(idx2)
                            missing.remove(idx2)
                            weights[idx2] *= 1.03
                            if current_count > best_local_count:
                                best_local_count = current_count
                                best_local_presence = hg.presence
                                best_local_common = [row[:] for row in hg.common]
                        else:
                            hg.remove(idx2)
                            hg.add(idx1)
                    else:
                        hg.add(idx1)
                        for cf in _:
                            weights[cf] *= 0.96
            T *= cooling
        
        # Restore best local solution
        hg.presence = best_local_presence
        hg.count = best_local_count
        hg.common = best_local_common
        
        # Final improvement: try to add any missing triples
        present_mask = hg.presence
        for idx in range(N_TRIPLES):
            if time.time() > deadline - 5:
                break
            if (present_mask >> idx) & 1:
                continue
            a, b, c = ALL_TRIPLES[idx]
            conflict, _ = hg.check_conflict(a, b, c)
            if not conflict:
                hg.add(idx)
                present_mask |= (1 << idx)
        
        if hg.count > best_count:
            best_count = hg.count
            best_hg = hg
            # Normalize weights
            max_w = max(weights) if weights else 1.0
            if max_w > 0:
                weights = [w / max_w for w in weights]
        
        # Decay weights for next restart
        weights = [w * 0.9 for w in weights]
    
    if best_hg is None:
        return []
        
    result = []
    for i in range(N_TRIPLES):
        if (best_hg.presence >> i) & 1:
            result.append(list(ALL_TRIPLES[i]))
            
    return result

random.seed(42)
