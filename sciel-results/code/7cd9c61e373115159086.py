def _compute_autoconv(x):
    n_fft = 1 << (2 * N - 1).bit_length()
    fx = np.fft.rfft(x, n_fft)
    ac_full = np.fft.irfft(fx * np.conj(fx), n_fft)
    return ac_full[:N]
