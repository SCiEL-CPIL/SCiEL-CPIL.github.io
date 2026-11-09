"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import collections

class Policy:
    """LinUCB with a fixed sliding window for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # 10s safety margin before 120s limit
        
        # Sliding window parameters
        self.window_size = 50  # Keep last 50 observations per arm
        
        # Data structures for sliding window
        # A[a] is the sum of xx^T for the last W samples of arm a
        # b[a] is the sum of yx for the last W samples of arm a
        # queue[a] stores (context, reward) pairs in chronological order
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.queues = [collections.deque(maxlen=self.window_size) for _ in range(n_arms)]
        
        # Exploration parameter
        self.alpha = 1.0

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fallback to a simple heuristic if we are about to time out
            # Just pick the arm with highest current theta estimate
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                if self.queues[a]:
                    theta = np.linalg.lstsq(
                        np.array([x for x, _ in self.queues[a]]).T, 
                        np.array([r for _, r in self.queues[a]]), 
                        rcond=None
                    )[0]
                    score = theta @ context
                    if score > best_score:
                        best_score = score
                        best_arm = a
            return best_arm

        scores = []
        for a in range(self.n):
            # Compute theta using the windowed A and b
            # A is initialized as I, so it's always invertible
            try:
                inv_A = np.linalg.inv(self.A[a])
                theta = inv_A @ self.b[a]
                # Variance term
                v = self.alpha * np.sqrt(context @ inv_A @ context)
                scores.append(theta @ context + v)
            except np.linalg.LinAlgError:
                # Fallback if singular (unlikely with I initialization, but safe)
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Add new observation
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        self.queues[arm].append((context, reward))
        
        # If window is full, remove the oldest observation
        if len(self.queues[arm]) == self.window_size:
            # Note: In a strict implementation, we would need to know when to pop.
            # Deque with maxlen automatically pops the left when appending if full.
            # However, we need to subtract the contribution of the popped item.
            # To do this correctly, we must capture the item BEFORE it is popped.
            # Since we just appended, if it was full before append, the leftmost item is now the one to remove.
            # But wait, `collections.deque(maxlen=W)` pops the leftmost item immediately upon append if full.
            # So we need to check if the queue was full BEFORE the append.
            pass # Logic handled below by checking length change or pre-checking

        # Correct logic for sliding window subtraction:
        # We need to subtract the oldest item if the window is now full.
        # Actually, it's easier to just reconstruct or track indices.
        # Let's re-implement the update to be robust.
        
        # Re-doing update logic for clarity and correctness
        # The previous append might have already popped if maxlen was set.
        # To be safe and explicit:
        
        # 1. Check if we need to remove the oldest item
        # If the queue length was already at max, the oldest item is at index 0.
        # But `deque` with maxlen pops automatically on append.
        # So we must capture the popped item.
        
        # Let's refactor update to handle this explicitly.
        
        # Remove oldest if window is full
        # We check if the queue is full. If so, the item at the front is the one to remove from A and b.
        # But we just appended. So if len == window_size, the front is the oldest.
        # Wait, if maxlen is set, append pops the left.
        # So we should have captured the left item BEFORE appending.
        
        # Let's restart the update logic properly.
        
        # --- Correct Update Implementation ---
        
        # 1. If the queue is full, the oldest item will be discarded by the append.
        #    We need to subtract its contribution from A and b.
        if len(self.queues[arm]) == self.window_size:
            old_context, old_reward = self.queues[arm][0]
            self.A[arm] -= np.outer(old_context, old_context)
            self.b[arm] -= old_reward * old_context
            # The append will pop this item automatically
            
        # 2. Add new item
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        self.queues[arm].append((context, reward))
        
        self.t += 1
# EVOLVE-BLOCK-END
