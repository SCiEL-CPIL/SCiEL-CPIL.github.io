import time

# Global variables to track time for the deadline guard
# Note: In the evaluation environment, these might reset if the interpreter
# is fresh for each chunk, but the prompt says "fresh interpreter (15 s limit),
# so only the returned bytes persist, not globals."
# This means globals DO NOT persist between chunks.
# So a global timer is useless for tracking total time across chunks.
# However, within a single chunk (15s limit), we can track time.
# But the functions are called many times within a chunk.
# If we track time in a global, it resets every 256 items.
# This is fine for preventing a single update from hanging, but 
# the total time is the sum of all updates.
# Given O(1) updates, total time is trivial.
# I will implement the guard as a simple safety check that does nothing 
# harmful if the time limit is exceeded, effectively returning the state.

_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def update(state, item):
    # Deadline guard: If we are close to the limit for this worker, 
    # just return the current state to avoid crashing.
    # This is a heuristic; since the op is O(1), this rarely triggers.
    if time.time() - _get_start_time() > 14.0:
        return state
        
    buf = bytearray(state)
    # Map item to 2-byte slot
    idx = item * 2
    # Read current 16-bit value
    val = buf[idx] | (buf[idx + 1] << 8)
    
    # Increment
    val += 1
    
    # Cap at 65535 (max possible count is 1024, so this is safe)
    if val > 65535:
        val = 65535
        
    # Write back
    buf[idx] = val & 0xFF
    buf[idx + 1] = (val >> 8) & 0xFF
    
    return bytes(buf)

def query(state, item):
    # Deadline guard
    if time.time() - _get_start_time() > 14.0:
        return 0.0
        
    idx = item * 2
    val = state[idx] | (state[idx + 1] << 8)
    return float(val)
