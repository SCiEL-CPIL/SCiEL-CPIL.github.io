value_next = np.zeros(n_nodes)
# ... initialize ...
for j in range(n_steps - 1, -1, -1):
    value_j = disc * (p * value_next[1:j+1] + q * value_next[0:j])
    # ... exercise check ...
    value_next[j+1:] = 0  # Clear unused entries
    value_next[0:j+1] = value_j
