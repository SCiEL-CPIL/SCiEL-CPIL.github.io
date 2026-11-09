import struct

def update(state, item):
    ba = bytearray(state)
    item = int(item)
    
    if item < 128:
        # Exact counting with uint16
        offset = item * 2
        val = struct.unpack_from('<H', ba, offset)[0]
        val += 1
        struct.pack_into('<H', ba, offset, val)
    else:
        # Count-Min Sketch for items 128-399
        # 4 rows, 192 slots each. uint8 counters.
        # Offset for row r: 256 + r * 192
        # Hash: ((item * P + row * Q) % LARGE) % 192
        
        # Primes for hashing
        p1 = 1035493289
        p2 = 775374203
        p3 = 1495862053
        p4 = 2015145641
        
        # We need 4 hashes.
        # Using a single large prime modulo might be faster than 4 large multiplies?
        # But we have 4 rows.
        
        # Let's use the provided primes from the seed, adapted for 192 slots.
        # The seed used modulo 64. We use modulo 192.
        
        # Row 0
        idx0 = (item * p1) % 192
        off0 = 256 + idx0
        ba[off0] = (ba[off0] + 1) % 256 # uint8 increment
        
        # Row 1
        idx1 = (item * p2) % 192
        off1 = 256 + 192 + idx1
        ba[off1] = (ba[off1] + 1) % 256
        
        # Row 2
        idx2 = (item * p3) % 192
        off2 = 256 + 192*2 + idx2
        ba[off2] = (ba[off2] + 1) % 256
        
        # Row 3
        idx3 = (item * p4) % 192
        off3 = 256 + 192*3 + idx3
        ba[off3] = (ba[off3] + 1) % 256
        
    return bytes(ba)

def query(state, item):
    item = int(item)
    
    if item < 128:
        offset = item * 2
        val = struct.unpack_from('<H', state, offset)[0]
        return float(val)
    else:
        p1 = 1035493289
        p2 = 775374203
        p3 = 1495862053
        p4 = 2015145641
        
        idx0 = (item * p1) % 192
        v0 = state[256 + idx0]
        
        idx1 = (item * p2) % 192
        v1 = state[256 + 192 + idx1]
        
        idx2 = (item * p3) % 192
        v2 = state[256 + 192*2 + idx2]
        
        idx3 = (item * p4) % 192
        v3 = state[256 + 192*3 + idx3]
        
        return float(min(v0, v1, v2, v3))
