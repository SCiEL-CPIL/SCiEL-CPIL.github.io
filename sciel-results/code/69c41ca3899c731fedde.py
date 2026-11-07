import time

DEADLINE = time.time() + 50.0

def construct():
    # ... setup ...
    
    def can_partition(rem, k):
        if time.time() > DEADLINE:
            return False # Or raise? False is safer to return a suboptimal result? 
            # But if we return False, we might increase k.
            # If we exceed deadline, we should probably return the best found so far?
            # But we are searching for min k.
            # If we time out, we don't have a valid partition?
            # The task says "Any broken rule makes the whole result invalid".
            # So we MUST return a valid partition.
            # If we time out, we should fall back to a greedy solution or singletons.
            # Singletons are always valid (clique of size 1).
            # So if we fail, return [[i] for i in range(n)].
            
        # ... logic ...
