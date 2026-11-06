"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys

_DEADLINE = None
_DEADLINE_MARGIN = 5.0  # seconds to keep before hard limit


def _setup_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 235.0  # 235s budget, leaving 5s margin


def _time_left():
    return _DEADLINE - time.time()


def _neighbours(point):
    """Yield 6 neighbors on cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy: -number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _beam_search(sequence, dimension, rng, beam_width=50, max_time=10.0):
    """Beam search with dynamic width and improved heuristic."""
    start = (0,) * dimension
    # Beam entry: (chain_list, occupied_dict, current_energy)
    beams = [([start], {start: 0}, 0)]
    n = len(sequence)
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    t0 = time.time()
    for i in range(1, n):
        if time.time() - t0 > max_time:
            break
        
        new_beams = []
        for chain, occupied, energy in beams:
            last = chain[-1]
            options = []
            for site in _neighbours(last):
                if site in occupied:
                    continue
                gain = 0
                if is_h[i]:
                    for q in _neighbours(site):
                        if q in occupied:
                            idx = occupied[q]
                            if idx < i - 1 and is_h[idx]:
                                gain += 1
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                # Look-ahead: count H residues in next few positions that could contact
                lookahead = 0
                if is_h[i]:
                    for j in range(i + 2, min(i + 5, n)):
                        if is_h[j]:
                            lookahead += 1
                # Heuristic: prefer high gain, then free space, then lookahead potential
                score = gain * 100 + free * 10 + lookahead * 5 + rng.random()
                options.append((score, site, gain))
            
            if not options:
                continue
            
            options.sort(key=lambda x: -x[0])
            # Keep top candidates per beam
            keep = min(beam_width, max(1, beam_width // 2))
            for score, site, gain in options[:keep]:
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                new_energy = energy - gain
                new_beams.append((new_chain, new_occupied, new_energy))
        
        if not new_beams:
            break
        
        # Keep top beam_width beams
        new_beams.sort(key=lambda x: x[2])
        beams = new_beams[:beam_width]
    
    if not beams:
        return None
    
    best = min(beams, key=lambda x: x[2])
    return best[0]


def _apply_pivot_move(chain, k, axis, direction, dimension):
    """
    Apply a pivot move: rotate segment k..n-1 around chain[k-1].
    Returns new chain or None if invalid (self-intersection).
    k is 1..n-1, pivot is at index k-1.
    """
    n = len(chain)
    if k < 1 or k >= n:
        return None
    
    pivot = chain[k-1]
    tail = chain[k:]
    
    # Build set of head positions for collision check
    head_set = set(chain[:k])
    
    # Rotate tail around pivot
    new_tail = []
    for p in tail:
        v = [p[d] - pivot[d] for d in range(dimension)]
        if axis == 0:
            if direction == 1:
                v_new = [v[0], -v[2], v[1]]
            else:
                v_new = [v[0], v[2], -v[1]]
        elif axis == 1:
            if direction == 1:
                v_new = [v[2], v[1], -v[0]]
            else:
                v_new = [-v[2], v[1], v[0]]
        else:
            if direction == 1:
                v_new = [-v[1], v[0], v[2]]
            else:
                v_new = [v[1], -v[0], v[2]]
        new_p = tuple(pivot[d] + v_new[d] for d in range(dimension))
        new_tail.append(new_p)
    
    # Check for self-intersection with head
    for p in new_tail:
        if p in head_set:
            return None
    
    return chain[:k] + new_tail


def _compute_delta_energy_pivot(sequence, chain, k, new_chain):
    """
    Compute energy difference when replacing tail (k..n-1) with new_tail.
    Only cross-contacts between head (0..k-1) and tail change.
    """
    n = len(chain)
    if k < 1 or k >= n:
        return 0
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    # Build position maps
    head = chain[:k]
    old_tail = chain[k:]
    new_tail = new_chain[k:]
    
    head_pos_set = set(head)
    old_tail_pos_set = set(old_tail)
    new_tail_pos_set = set(new_tail)
    
    # Map positions to indices in head and tails
    head_idx = {p: i for i, p in enumerate(head)}
    old_tail_idx = {p: i for i, p in enumerate(old_tail)}
    new_tail_idx = {p: i for i, p in enumerate(new_tail)}
    
    # Count old cross-contacts: H in head adjacent to H in old_tail
    old_cross = 0
    for i, p in enumerate(head):
        if not is_h[i]:
            continue
        for nb in _neighbours(p):
            if nb in old_tail_pos_set:
                j = old_tail_idx[nb]
                if is_h[k + j]:
                    old_cross += 1
    
    # Count new cross-contacts
    new_cross = 0
    for i, p in enumerate(head):
        if not is_h[i]:
            continue
        for nb in _neighbours(p):
            if nb in new_tail_pos_set:
                j = new_tail_idx[nb]
                if is_h[k + j]:
                    new_cross += 1
    
    # Energy = -contacts, so delta = -(new_cross - old_cross) = old_cross - new_cross
    return old_cross - new_cross


class _FastLocalSearcher:
    def __init__(self, sequence, chain, seed=42):
        self.sequence = sequence
        self.chain = list(chain)
        self.n = len(self.chain)
        self.is_h = [1 if s == 'H' else 0 for s in sequence]
        self.rng = random.Random(seed)
        self.occupied = {p: i for i, p in enumerate(self.chain)}
        self.contacts = self._init_contacts()
        self.energy = -len(self.contacts)

    def _init_contacts(self):
        contacts = set()
        for i in range(self.n):
            if self.is_h[i]:
                p = self.chain[i]
                for q in _neighbours(p):
                    j = self.occupied.get(q)
                    if j is not None and j > i + 1 and self.is_h[j]:
                        contacts.add((i, j))
        return contacts

    def _remove_contacts_for(self, idx):
        """Remove all contacts involving idx. Returns number of contacts removed."""
        removed = 0
        to_remove = []
        for (i, j) in self.contacts:
            if i == idx or j == idx:
                to_remove.append((i, j))
        for c in to_remove:
            self.contacts.remove(c)
            removed += 1
        return removed

    def _add_contacts_for(self, idx):
        """Add all new contacts involving idx. Returns number of contacts added."""
        added = 0
        p = self.chain[idx]
        for q in _neighbours(p):
            j = self.occupied.get(q)
            if j is not None and j > idx + 1 and self.is_h[j]:
                pair = (idx, j)
                if pair not in self.contacts:
                    self.contacts.add(pair)
                    added += 1
        return added

    def _try_move_180(self, i):
        """180-degree turn at i (moves i-1, i+1)."""
        if i < 1 or i > self.n - 2:
            return False
        a, b, c = self.chain[i-1], self.chain[i], self.chain[i+1]
        new_a = (2*b[0]-c[0], 2*b[1]-c[1], 2*b[2]-c[2])
        new_c = (2*b[0]-a[0], 2*b[1]-a[1], 2*b[2]-a[2])
        
        if new_a == new_c or new_a == b or new_c == b:
            return False
        if new_a in self.occupied and self.occupied[new_a] != i-1:
            return False
        if new_c in self.occupied and self.occupied[new_c] != i+1:
            return False
            
        # Update
        self._remove_contacts_for(i-1)
        self._remove_contacts_for(i+1)
        
        self.occupied.pop(a)
        self.occupied.pop(c)
        self.chain[i-1] = new_a
        self.chain[i+1] = new_c
        self.occupied[new_a] = i-1
        self.occupied[new_c] = i+1
        
        added = self._add_contacts_for(i-1) + self._add_contacts_for(i+1)
        # Check contact between i-1 and i+1
        if self.is_h[i-1] and self.is_h[i+1] and abs((i-1)-(i+1)) > 1:
            if abs(new_a[0]-new_c[0]) + abs(new_a[1]-new_c[1]) + abs(new_a[2]-new_c[2]) == 1:
                if (i-1, i+1) not in self.contacts:
                    self.contacts.add((i-1, i+1))
                    added += 1
        
        self.energy -= added
        return True

    def _try_move_3shift(self, i):
        """3-residue shift at i (moves i-1, i, i+1)."""
        if i < 1 or i > self.n - 2:
            return False
        a, b, c = self.chain[i-1], self.chain[i], self.chain[i+1]
        
        for axis in range(3):
            for step in (-1, 1):
                na = list(a); na[axis] += step; na = tuple(na)
                nb = list(b); nb[axis] += step; nb = tuple(nb)
                nc = list(c); nc[axis] += step; nc = tuple(nc)
                
                if na in self.occupied and self.occupied[na] != i-1: continue
                if nb in self.occupied and self.occupied[nb] != i: continue
                if nc in self.occupied and self.occupied[nc] != i+1: continue
                
                # Check adjacency
                if abs(na[0]-nb[0]) + abs(na[1]-nb[1]) + abs(na[2]-nb[2]) != 1: continue
                if abs(nb[0]-nc[0]) + abs(nb[1]-nc[1]) + abs(nb[2]-nc[2]) != 1: continue
                
                # Update
                self._remove_contacts_for(i-1)
                self._remove_contacts_for(i)
                self._remove_contacts_for(i+1)
                
                self.occupied.pop(a)
                self.occupied.pop(b)
                self.occupied.pop(c)
                self.chain[i-1] = na
                self.chain[i] = nb
                self.chain[i+1] = nc
                self.occupied[na] = i-1
                self.occupied[nb] = i
                self.occupied[nc] = i+1
                
                added = self._add_contacts_for(i-1) + self._add_contacts_for(i) + self._add_contacts_for(i+1)
                
                # Check internal contacts
                if self.is_h[i-1] and self.is_h[i+1] and abs((i-1)-(i+1)) > 1:
                    if abs(na[0]-nc[0]) + abs(na[1]-nc[1]) + abs(na[2]-nc[2]) == 1:
                        if (i-1, i+1) not in self.contacts:
                            self.contacts.add((i-1, i+1))
                            added += 1
                            
                self.energy -= added
                return True
        return False

    def _try_move_pivot(self, i):
        """Pivot move at bond i-i+1 (moves i+1)."""
        if i < 1 or i > self.n - 3:
            return False
        a, b = self.chain[i], self.chain[i+1]
        
        for axis in range(3):
            for step in (-1, 1):
                nb = list(b); nb[axis] += step; nb = tuple(nb)
                if nb == b: continue
                if nb in self.occupied and self.occupied[nb] != i+1: continue
                if abs(a[0]-nb[0]) + abs(a[1]-nb[1]) + abs(a[2]-nb[2]) != 1: continue
                if i + 2 < self.n:
                    if abs(nb[0]-self.chain[i+2][0]) + abs(nb[1]-self.chain[i+2][1]) + abs(nb[2]-self.chain[i+2][2]) != 1:
                        continue
                
                # Update
                self._remove_contacts_for(i+1)
                
                self.occupied.pop(b)
                self.chain[i+1] = nb
                self.occupied[nb] = i+1
                
                added = self._add_contacts_for(i+1)
                self.energy -= added
                return True
        return False

    def _try_move_crankshaft(self, i):
        """Crankshaft move at i (moves i+1, i+2)."""
        if i < 1 or i > self.n - 4:
            return False
        a, b, c = self.chain[i], self.chain[i+1], self.chain[i+2]
        
        # Try rotating (b, c) around a
        # There are 3 axes of rotation around a.
        # For each axis, there are 2 directions (60, 300) or just 180? 
        # In lattice, we can try all valid new positions for b and c that maintain connectivity.
        
        # A simpler crankshaft: move b and c such that a-b-c is preserved but shifted?
        # No, standard crankshaft rotates the segment.
        
        # Let's try a simpler "2-shift" or "crank" that moves i+1 and i+2.
        # We will try to move i+1 to a new position nb, and i+2 to a new position nc.
        
        candidates = []
        for nb in _neighbours(a):
            if nb == b: continue
            if nb in self.occupied and self.occupied[nb] != i+1: continue
            for nc in _neighbours(nb):
                if nc == c: continue
                if nc in self.occupied and self.occupied[nc] != i+2: continue
                # Check if nc is adjacent to i+3
                if i + 3 < self.n:
                    if abs(nc[0]-self.chain[i+3][0]) + abs(nc[1]-self.chain[i+3][1]) + abs(nc[2]-self.chain[i+3][2]) != 1:
                        continue
                candidates.append((nb, nc))
        
        if not candidates:
            return False
            
        # Pick a random candidate (or the best if we compute delta, but here we just try one)
        nb, nc = self.rng.choice(candidates)
        
        # Update
        self._remove_contacts_for(i+1)
        self._remove_contacts_for(i+2)
        
        self.occupied.pop(b)
        self.occupied.pop(c)
        self.chain[i+1] = nb
        self.chain[i+2] = nc
        self.occupied[nb] = i+1
        self.occupied[nc] = i+2
        
        added = self._add_contacts_for(i+1) + self._add_contacts_for(i+2)
        
        # Check contact between i+1 and i+2 (adjacent in chain, so no contact)
        # Check contact between i+1 and i+2 if they are non-adjacent? No, they are adjacent.
        
        self.energy -= added
        return True

    def run(self, max_iters=1000):
        improved = True
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            if _time_left() < 1.0:
                break
            for _ in range(50):
                i = self.rng.randint(1, self.n - 2)
                move_type = self.rng.random()
                if move_type < 0.4:
                    if self._try_move_180(i): improved = True
                elif move_type < 0.7:
                    if self._try_move_3shift(i): improved = True
                elif move_type < 0.85:
                    if self._try_move_pivot(i): improved = True
                else:
                    if self._try_move_crankshaft(i): improved = True
        return self.chain, self.energy


def _local_search(sequence, chain, max_iters=2000, seed=42):
    """Improve a conformation via 180-degree turns, 3-residue shifts, and pivot moves with incremental energy."""
    n = len(chain)
    if n < 4:
        return chain
    
    searcher = _FastLocalSearcher(sequence, chain, seed=seed)
    chain_out, _ = searcher.run(max_iters=max_iters)
    return chain_out


def _sa_pivot_search(sequence, chain, dimension, rng, max_time, is_h):
    """
    Pivot-move local search with simulated annealing.
    Returns best chain found.
    """
    n = len(chain)
    if n < 2:
        return chain
    
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T_max = 8.0
    T_min = 0.05
    
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        if _time_left() < 2.0:
            break
        
        # Choose pivot point k (1 to n-1)
        k = rng.randint(1, n - 1)
        axis = rng.randint(0, dimension - 1)
        direction = rng.choice([-1, 1])
        
        new_chain = _apply_pivot_move(current_chain, k, axis, direction, dimension)
        if new_chain is None:
            continue
        
        delta = _compute_delta_energy_pivot(sequence, current_chain, k, new_chain)
        
        # Metropolis acceptance
        if delta < 0:
            accept = True
        else:
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / max_time)
            T = T_max - (T_max - T_min) * progress
            if T < T_min:
                T = T_min
            # Use math.exp for accuracy
            prob = math.exp(-delta / T)
            accept = rng.random() < prob
        
        if accept:
            current_chain = new_chain
            current_energy += delta
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
    
    return best_chain


def _spiral_fold(sequence, dimension, seed=0):
    """Create a spiral-like initial conformation."""
    n = len(sequence)
    rng = random.Random(seed)
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
    dir_idx = rng.randint(0, 5)
    
    for i in range(1, n):
        placed = False
        # Try preferred direction first, then others
        order = [dir_idx % 6] + [d for d in range(6) if d != dir_idx % 6]
        rng.shuffle(order[1:])
        for d_idx in order:
            d = directions[d_idx % 6]
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                placed = True
                # Occasionally turn
                if rng.random() < 0.3:
                    dir_idx = (d_idx + 1) % 6
                break
        
        if not placed:
            # Stuck, try all 6 directions
            for d in directions:
                new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied[new_pos] = i
                    break
            else:
                break
    
    return chain


def fold(sequence, dimension):
    """Fold a sequence on the cubic lattice using multiple strategies."""
    _setup_deadline()
    n = len(sequence)
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    # Strategy 1: Multiple spiral initializations
    for seed in range(5):
        if _time_left() < 3.0:
            break
        chain1 = _spiral_fold(sequence, dimension, seed=seed)
        if len(chain1) == n:
            e1 = _compute_energy(sequence, chain1)
            if best_chain is None or e1 < best_energy:
                best_chain = chain1
                best_energy = e1
    
    # Strategy 2: Beam search with different widths
    if _time_left() > 5.0:
        for bw in [30, 80, 150]:
            if _time_left() < 3.0:
                break
            max_t = min(8.0, _time_left() * 0.25)
            chain2 = _beam_search(sequence, dimension, rng, beam_width=bw, max_time=max_t)
            if chain2 and len(chain2) == n:
                e2 = _compute_energy(sequence, chain2)
                if best_chain is None or e2 < best_energy:
                    best_chain = chain2
                    best_energy = e2
    
    # Strategy 3: Local search refinement on best chain
    if best_chain is not None and _time_left() > 2.0:
        # Run local search multiple times with different seeds
        for ls_seed in range(8):
            if _time_left() < 2.0:
                break
            max_iters = min(6000, int(_time_left() * 150))
            refined = _local_search(sequence, best_chain, max_iters=max_iters, seed=ls_seed * 7 + n)
            if refined and len(refined) == n:
                er = _compute_energy(sequence, refined)
                if er < best_energy:
                    best_chain = refined
                    best_energy = er
    
    # Strategy 4: Simulated annealing with pivot moves
    if best_chain is not None and _time_left() > 3.0:
        sa_time = min(10.0, _time_left() * 0.4)
        sa_refined = _sa_pivot_search(sequence, best_chain, dimension, rng, sa_time, is_h)
        if sa_refined and len(sa_refined) == n:
            e_sa = _compute_energy(sequence, sa_refined)
            if e_sa < best_energy:
                best_chain = sa_refined
                best_energy = e_sa
    
    # Final polish: one more round of local search on the best found
    if best_chain is not None and _time_left() > 2.0:
        final_refined = _local_search(sequence, best_chain, max_iters=2000, seed=n + 100)
        if final_refined and len(final_refined) == n:
            e_final = _compute_energy(sequence, final_refined)
            if e_final < best_energy:
                best_chain = final_refined
                best_energy = e_final
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
