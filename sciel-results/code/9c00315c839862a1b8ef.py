    import random
    import time
    import itertools
    import math

    def construct(v, k, t):
        # Precompute triples
        all_subs = list(itertools.combinations(range(v), t))
        sub_to_idx = {s: i for i, s in enumerate(all_subs)}
        n_subs = len(all_subs)
        
        # Helper to get mask of a block
        def get_block_mask(block):
            mask = 0
            for tr in itertools.combinations(block, t):
                idx = sub_to_idx[tr]
                mask |= (1 << idx)
            return mask

        # Initial state: Random blocks
        def init_state(num_blocks):
            blocks = []
            for _ in range(num_blocks):
                b = sorted(random.sample(range(v), k))
                blocks.append(b)
            return blocks

        def calc_energy(blocks):
            cover_counts = [0] * n_subs
            for b in blocks:
                mask = get_block_mask(b)
                # This is inefficient to recalculate from scratch every time
                # Better to maintain state
            pass

        # State object
        class State:
            def __init__(self, blocks):
                self.blocks = blocks
                self.cover_counts = [0] * n_subs
                self.uncovered_mask = (1 << n_subs) - 1
                for b in blocks:
                    self.add_block(b)
                
            def add_block(self, b):
                mask = get_block_mask(b)
                # Update counts
                for tr in itertools.combinations(b, t):
                    idx = sub_to_idx[tr]
                    if self.cover_counts[idx] == 0:
                        self.uncovered_mask &= ~(1 << idx)
                    self.cover_counts[idx] += 1
                self.blocks.append(b)
                
            def remove_block(self, b):
                mask = get_block_mask(b)
                for tr in itertools.combinations(b, t):
                    idx = sub_to_idx[tr]
                    self.cover_counts[idx] -= 1
                    if self.cover_counts[idx] == 0:
                        self.uncovered_mask |= (1 << idx)
                self.blocks.remove(b)
                
            def get_energy(self):
                return self.uncovered_mask.bit_count() * 100 + len(self.blocks)
                
            def copy(self):
                # Deep copy
                new_state = State([])
                new_state.blocks = self.blocks[:]
                new_state.cover_counts = self.cover_counts[:]
                new_state.uncovered_mask = self.uncovered_mask
                return new_state

        # SA Loop
        start_time = time.time()
        time_limit = 110
        
        # Start with a greedy solution to give SA a good head start?
        # The directive says "do not reuse current designs".
        # Using Greedy to *initialize* SA is a hybrid.
        # I will start with a random state to be safe, or a very simple greedy.
        # Let's use a simple greedy to initialize, as pure random SA will take forever to find a valid covering.
        
        # Simple Greedy Init
        def greedy_init():
            uncovered = set(range(n_subs))
            blocks = []
            while uncovered:
                # Pick random uncovered
                tr_idx = random.choice(list(uncovered))
                tr = all_subs[tr_idx]
                # Find best block containing tr
                best_gain = -1
                best_block = None
                # Sample candidates
                candidates = []
                remaining = [p for p in range(v) if p not in tr]
                # Sample 100 random completions
                for _ in range(100):
                    extra = random.sample(remaining, k-t)
                    b = sorted(list(tr) + extra)
                    gain = 0
                    for tr2 in itertools.combinations(b, t):
                        if sub_to_idx[tr2] in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b
                blocks.append(best_block)
                for tr2 in itertools.combinations(best_block, t):
                    uncovered.discard(sub_to_idx[tr2])
            return blocks

        init_blocks = greedy_init()
        current_state = State(init_blocks)
        best_state = current_state.copy()
        best_energy = current_state.get_energy()
        
        T = 100.0
        T_min = 0.1
        alpha = 0.99
        
        while time.time() - start_time < time_limit:
            # Generate Move
            move_type = random.random()
            new_state = current_state.copy()
            
            if move_type < 0.3:
                # Add Block
                # Pick a block that covers many uncovered triples
                # Or random
                if current_state.uncovered_mask:
                    # Pick random uncovered triple
                    # Get index from mask
                    # This is hard with int mask.
                    # Just pick random block
                    b = sorted(random.sample(range(v), k))
                    new_state.add_block(b)
                else:
                    # If valid, try to remove
                    i = random.randint(0, len(new_state.blocks)-1)
                    b = new_state.blocks[i]
                    new_state.remove_block(b)
            elif move_type < 0.6:
                # Remove Block
                if new_state.blocks:
                    i = random.randint(0, len(new_state.blocks)-1)
                    b = new_state.blocks[i]
                    new_state.remove_block(b)
            else:
                # Swap
                if len(new_state.blocks) > 0:
                    i = random.randint(0, len(new_state.blocks)-1)
                    b_old = new_state.blocks[i]
                    b_new = sorted(random.sample(range(v), k))
                    new_state.remove_block(b_old)
                    new_state.add_block(b_new)
            
            new_energy = new_state.get_energy()
            delta_e = new_energy - current_state.get_energy()
            
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                current_state = new_state
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_state = new_state.copy()
            
            T *= alpha
            if T < T_min:
                T = T_min
                
        return best_state.blocks
    