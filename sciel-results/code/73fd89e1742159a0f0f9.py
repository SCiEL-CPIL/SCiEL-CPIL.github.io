<<<<<<< SEARCH
import numpy as np
import time

N = 400
DEADLINE = 140.0  # Safety margin before 150s kill

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound: max |autoconv(t)| for the signed step function.
    """
    M = 2 * N  # fine grid for accuracy
    dx = 0.5 / M
    
    # Build fine grid
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N - 1)
    f_fine = signed_step.astype(np.float64)[i_indices]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2 * M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    return c3
=======
import numpy as np
import time

N = 400
DEADLINE = 140.0  # Safety margin before 150s kill

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound: max |autoconv(t)| for the signed step function.
    """
    M = 2 * N  # fine grid for accuracy
    dx = 0.5 / M
    
    # Build fine grid
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N - 1)
    f_fine = signed_step.astype(np.float64)[i_indices]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2 * M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    return c3

def _build_fine_grid_indices():
    """Precompute indices for mapping coarse to fine grid."""
    M = 2 * N
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N - 1)
    return M, i_indices

def _compute_autoconv_from_fft(fft_f, M):
    """Compute autoconvolution from FFT."""
    dx = 0.5 / M
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    return autoconv

def _get_c3_from_autoconv(autoconv):
    """Get C3 value from autoconvolution."""
    return np.max(np.abs(autoconv))
>>>>>>> REPLACE

