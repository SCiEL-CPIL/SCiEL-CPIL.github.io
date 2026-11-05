    current_state = np.asarray(current_state, dtype=float)
    if current_state.ndim == 0:
        current_state = current_state.reshape(1)
    elif current_state.ndim > 1:
        current_state = current_state.flatten()
    