import time
start_time = time.time()
# ... compute ...
if time.time() - start_time > 115:
    # Fallback? No, it's already done.
    pass
return ...
