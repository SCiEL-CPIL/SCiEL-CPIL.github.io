"""HP lattice protein folding using fast simulated annealing with incremental energy updates."""
# EVOLVE-BLOCK-START
import random
import math
import time


def _neighbours(point):
    """Yield 4-neighbours of a point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _build_initial_compact(n):
    """
    Create a compact initial self-avoiding walk.
    Uses a spiral pattern.
    """
    if n <= 1:
        return [(0, 0)]
    
    coords = []
    x, y = 0, 0
    dx, dy = 1, 0
    steps_in_dir = 1
    dir_changes = 0
    
    while len(coords) < n:
        coords.append((x, y))
        if len(coords) == n:
            break
        x += dx
        y += dy
        dir_changes += 1
        if dir_changes == steps_in_dir:
            # Rotate direction
            dx, dy = -dy, dx
            dir_changes = 0
            if dir_changes % 2 == 0: # Every 2 turns increase step length
                steps_in_dir += 1
                
    return coords


class HPFolder:
    def __init__(self, sequence, initial_chain=None):
        self.sequence = sequence
        self.n = len(sequence)
        self.is_h = [c == 'H' for c in sequence]
        
        if initial_chain is None:
            self.chain = _build_initial_compact(self.n)
        else:
            self.chain = initial_chain
            
        self._init_state()

    def _init_state(self):
        """Initialize data structures for the current chain."""
        self.pos_to_idx = {p: i for i, p in enumerate(self.chain)}
        self.contacts = set()
        
        # Compute initial contacts
        for i in range(self.n):
            if not self.is_h[i]:
                continue
            x, y = self.chain[i]
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                j = self.pos_to_idx.get((nx, ny))
                if j is not None and abs(i - j) > 1 and self.is_h[j]:
                    if i < j:
                        self.contacts.add((i, j))
                    else:
                        self.contacts.add((j, i))
                        
        self.energy = -len(self.contacts)

    def _try_reversal(self, i, j, rng, T):
        """
        Attempt to reverse subchain [i, j].
        Returns True if move was accepted.
        """
        if i >= j:
            return False
            
        # Check boundary connectivity
        if i > 0:
            p_prev = self.chain[i-1]
            p_new_i = self.chain[j]
            if abs(p_prev[0] - p_new_i[0]) + abs(p_prev[1] - p_new_i[1]) != 1:
                return False
        if j < self.n - 1:
            p_next = self.chain[j+1]
            p_new_j = self.chain[i]
            if abs(p_next[0] - p_new_j[0]) + abs(p_next[1] - p_new_j[1]) != 1:
                return False
            
        # Calculate Energy Delta
        old_energy = self.energy
        
        # 1. Remove all contacts involving any index in [i, j]
        affected_indices = set(range(i, j+1))
        contacts_to_remove = set()
        for c in self.contacts:
            if c[0] in affected_indices or c[1] in affected_indices:
                contacts_to_remove.add(c)
        
        for c in contacts_to_remove:
            self.contacts.remove(c)
        
        # 2. Update Chain and PosToIdx
        # Reverse the subchain
        self.chain[i:j+1] = self.chain[i:j+1][::-1]
        
        # Update pos_to_idx for the moved indices
        for k in range(i, j+1):
            self.pos_to_idx[self.chain[k]] = k
            
        # 3. Add new contacts for affected indices
        for k in range(i, j+1):
            if not self.is_h[k]:
                continue
            x, y = self.chain[k]
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                j_idx = self.pos_to_idx.get((nx, ny))
                if j_idx is not None and abs(k - j_idx) > 1 and self.is_h[j_idx]:
                    if k < j_idx:
                        self.contacts.add((k, j_idx))
                    else:
                        self.contacts.add((j_idx, k))
                        
        new_energy = -len(self.contacts)
        delta = new_energy - old_energy
        
        # Metropolis acceptance
        if delta < 0 or rng.random() < math.exp(-delta / T):
            self.energy = new_energy
            return True
        else:
            # Revert
            self.chain[i:j+1] = self.chain[i:j+1][::-1]
            for k in range(i, j+1):
                self.pos_to_idx[self.chain[k]] = k
            self.energy = old_energy
            return False

    def _try_end_move(self, end_idx, rng, T):
        """
        Move the end residue (index 0 or n-1) to a new adjacent position.
        """
        if self.n < 2:
            return False
            
        # Determine neighbor index
        if end_idx == 0:
            neighbor_idx = 1
        else:
            neighbor_idx = self.n - 2
            
        p_end = self.chain[end_idx]
        p_neighbor = self.chain[neighbor_idx]
        
        # Find free neighbors of p_neighbor
        candidates = []
        for nx, ny in _neighbours(p_neighbor):
            if (nx, ny) == p_end:
                continue
            if (nx, ny) not in self.pos_to_idx:
                candidates.append((nx, ny))
                
        if not candidates:
            return False
            
        new_pos = rng.choice(candidates)
        
        # Calculate Energy Delta
        old_energy = self.energy
        
        # Remove contacts involving end_idx
        contacts_to_remove = set()
        for c in self.contacts:
            if c[0] == end_idx or c[1] == end_idx:
                contacts_to_remove.add(c)
        for c in contacts_to_remove:
            self.contacts.remove(c)
            
        # Update state
        old_pos = p_end
        self.chain[end_idx] = new_pos
        del self.pos_to_idx[old_pos]
        self.pos_to_idx[new_pos] = end_idx
        
        # Add new contacts for end_idx
        if self.is_h[end_idx]:
            x, y = new_pos
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                j_idx = self.pos_to_idx.get((nx, ny))
                if j_idx is not None and abs(end_idx - j_idx) > 1 and self.is_h[j_idx]:
                    if end_idx < j_idx:
                        self.contacts.add((end_idx, j_idx))
                    else:
                        self.contacts.add((j_idx, end_idx))
                        
        new_energy = -len(self.contacts)
        delta = new_energy - old_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            self.energy = new_energy
            return True
        else:
            # Revert
            self.chain[end_idx] = old_pos
            del self.pos_to_idx[new_pos]
            self.pos_to_idx[old_pos] = end_idx
            self.energy = old_energy
            return False
            
    def fold(self, time_budget, rng):
        """Run SA for the given time budget."""
        start_time = time.time()
        
        # SA Parameters
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        
        best_chain = [list(p) for p in self.chain]
        best_energy = self.energy
        
        while T > T_min:
            if time.time() - start_time > time_budget:
                break
                
            # Choose move
            r = rng.random()
            
            if r < 0.6:
                # Reversal
                i = rng.randint(0, self.n - 2)
                j = rng.randint(i + 1, self.n - 1)
                self._try_reversal(i, j, rng, T)
            else:
                # End Move
                end_idx = rng.choice([0, self.n - 1])
                self._try_end_move(end_idx, rng, T)
                
            T *= cooling
            
            # Update best
            if self.energy < best_energy:
                best_energy = self.energy
                best_chain = [list(p) for p in self.chain]
                
        return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using simulated annealing with multiple restarts."""
    n = len(sequence)
    if n <= 2:
        return [[0, 0], [1, 0]][:n]
    
    total_budget = 10.0
    start_time = time.time()
    
    best_chain = None
    best_energy = 0
    
    # Multiple restarts
    n_restarts = 3
    time_per_restart = total_budget / n_restarts
    
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        remaining = total_budget - elapsed
        if remaining < 0.5:
            break
            
        rng = random.Random(12345 + len(sequence) * 100 + restart * 7)
        current_budget = min(time_per_restart, remaining)
        
        if restart == 0:
            folder = HPFolder(sequence)
        else:
            # Continue from best chain found so far
            # We need to be careful: HPFolder modifies its internal state.
            # We should create a new folder with the best chain as initial.
            folder = HPFolder(sequence, initial_chain=best_chain)
            
        chain, energy = folder.fold(current_budget, rng)
        
        if chain is not None:
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy

    if best_chain is None:
        # Fallback
        best_chain = [[i, 0] for i in range(n)]
        
    return best_chain

# EVOLVE-BLOCK-END
