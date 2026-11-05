_START_TIME = time.time()
_TIME_LIMIT = 115.0  # 5s buffer before 120s limit

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to simple greedy if time is running out
        if len(remaining_weights) == 0: return -1
        densities = np.array(remaining_values) / np.array(remaining_weights)
        return int(np.argmax(densities))
    # ... DP logic ...
