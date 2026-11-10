"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Generate Cantor set numbers in [0, 511]
    # Base 3 digits 0 and 1, up to 6 digits (3^5 = 243, 3^6 = 729)
    # Max value with 6 digits (0/1) is 111111_3 = 364
    # So all 2^6 = 64 numbers are <= 511
    
    cantor_numbers = []
    for i in range(64):
        val = 0
        # Convert i to 6-bit binary, then interpret as base-3 digits
        for bit in range(6):
            if (i >> bit) & 1:
                val += (3 ** bit)
        cantor_numbers.append(val + 1) # Shift to 1..512
        
    # The set is distinct and within [1, 512]
    return cantor_numbers
# EVOLVE-BLOCK-END
